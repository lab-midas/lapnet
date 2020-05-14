import os
import sys
import shutil
import PIL
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from pyexcel_ods import get_data
import pylab
import operator
import time
from multiprocessing import Pool
from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct, flow_to_color_np
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.input import KITTIInput
from e2eflow.kitti.data import KITTIData
from e2eflow.kitti.input import MRI_Resp_2D, np_warp_2D
from e2eflow.core.supervised import supervised_loss
from e2eflow.core.input import resize_input, resize_output_crop, resize_output, resize_output_flow
from e2eflow.core.train import restore_networks
from e2eflow.core.input import load_mat_file
from e2eflow.gui import display
from e2eflow.util import convert_input_strings
from e2eflow.test.Warp_assessment3D import warp_assessment3D
from e2eflow.core.image_warp import np_warp_2D, np_warp_3D


tf.app.flags.DEFINE_string('dataset', 'resp_2D',
                            'Name of dataset to evaluate on. One of {kitti, sintel, chairs, mdb}.')
tf.app.flags.DEFINE_string('variant', 'train_2015',
                           'Name of variant to evaluate on.'
                           'If dataset = kitti, one of {train_2012, train_2015, test_2012, test_2015}.'
                           'If dataset = sintel, one of {train_clean, train_final}.'
                           'If dataset = mdb, one of {train, test}.')
tf.app.flags.DEFINE_string('ex', '',
                           'Experiment name(s) (can be comma separated list).')
tf.app.flags.DEFINE_integer('num', 1,
                            'Number of examples to evaluate. Set to -1 to evaluate all.')
tf.app.flags.DEFINE_integer('num_vis', 1,
                            'Number of evalutations to visualize. Set to -1 to visualize all.')
tf.app.flags.DEFINE_string('gpu', '1',
                           'GPU device to evaluate on.')
tf.app.flags.DEFINE_boolean('output_benchmark', False,
                            'Output raw flow files.')
tf.app.flags.DEFINE_boolean('output_visual', False,
                            'Output flow visualization files.')
tf.app.flags.DEFINE_boolean('output_backward', False,
                            'Output backward flow files.')
tf.app.flags.DEFINE_boolean('output_png', True, # TODO finish .flo output
                            'Raw output format to use with output_benchmark.'
                            'Outputs .png flow files if true, output .flo otherwise.')
FLAGS = tf.app.flags.FLAGS


NUM_EXAMPLES_PER_PAGE = 4

def cal_loss_mean(loss_dir):
    if os.path.exists(os.path.join(loss_dir, 'mean_loss.txt')):
        raise ImportError('mean value already calculated')
    files = os.listdir(loss_dir)
    files.sort()
    mean = dict()
    for file in files:
        with open(os.path.join(loss_dir, file), 'r') as f:
            data = [float(i) for i in f.readlines()]
            mean[file.split('.')[0]] = np.mean(data)
    with open(os.path.join(loss_dir, 'mean_loss.txt'), "a") as f:
        for name in mean:
            f.write('{}:{}\n'.format(name, round(mean[name], 5)))


def _evaluate_experiment(name, data, config):

    current_config = config_dict('../config.ini')
    exp_dir = os.path.join(current_config['dirs']['log'], 'ex', name)
    config_path = os.path.join(exp_dir, 'config.ini')
    if not os.path.isfile(config_path):
        config_path = '../config.ini'
    if not os.path.isdir(exp_dir) or not tf.train.get_checkpoint_state(exp_dir):
        exp_dir = os.path.join(current_config['dirs']['checkpoints'], name)
    config_train = config_dict(config_path)
    params = config_train['train']
    convert_input_strings(params, config_dict('../config.ini')['dirs'])
    dataset_params_name = 'train_' + FLAGS.dataset
    if dataset_params_name in config_train:
        params.update(config_train[dataset_params_name])
    ckpt = tf.train.get_checkpoint_state(exp_dir)
    if not ckpt:
        raise RuntimeError("Error: experiment must contain a checkpoint")
    ckpt_path = exp_dir + "/" + os.path.basename(ckpt.model_checkpoint_path)

    batch_size = config['batch_size']
    height = params['height']
    width = params['width']

    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        test_batch, im1, im2, flow_orig = data()

        loss, flow = supervised_loss(
                             test_batch,
                             normalization=None,
                             augment=False,
                             params=params)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restore_networks(sess, params, ckpt, ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,                                                   coord=coord)

            #flow_final = np.zeros((height, width, len(config['selected_slices'])), dtype=np.float32)
            flow_final, _ = sess.run([flow, loss])  # todo: doesn't work if slices > batch_size
            flow_final = flow_final[:len(config['selected_slices']), ...]
            coord.request_stop()
            coord.join(threads)

        u_GT = (flow_orig[..., 0], flow_orig[..., 1])  # tuple
        u_est = (flow_final[..., 0], flow_final[..., 1])  # tuple
        OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
        error_data_pred = warp_assessment3D(u_GT, u_est, OF_index)

        size_mtx = np.shape(flow_orig[..., 0])
        u_GT = (np.zeros(size_mtx, dtype=np.float32), np.zeros(size_mtx, dtype=np.float32))  # tuple
        u_est = (flow_orig[..., 0], flow_orig[..., 1])  # tuple
        OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
        error_data_gt = warp_assessment3D(u_GT, u_est, OF_index)

        final_loss_orig = error_data_gt['Abs_Error_mean']
        final_loss = error_data_pred['Abs_Error_mean']
        final_loss_orig_angel = error_data_gt['Angle_Error_Mean']
        final_loss_angel = error_data_pred['Angle_Error_Mean']
        # final_loss_orig = np.mean(np.sqrt(np.sum(np.square(flow_orig), 3)))
        # #final_loss_orig = np.mean(np.square(flow_orig))
        # final_loss = np.mean(np.sqrt(np.sum(np.square(flow_final-flow_orig), 3)))
        # #final_loss = np.mean(np.square(flow_final-flow_orig))

        im1_pred = [np_warp_2D(i, j) for i, j in zip(list(im2), list(-flow_final))]
        im1_gt = [np_warp_2D(i, j) for i, j in zip(list(im2), list(-flow_orig))]
        color_flow_final = [flow_to_color_np(i) for i in list(flow_final)]
        color_flow_gt = [flow_to_color_np(i) for i in list(flow_orig)]


        # flow_neg = list(-flow_final)
        # flow_gt_neg = list(-flow_orig)
        # flow_gt, flow_final, im1, im2 = list(flow_orig), list(flow_final), list(im1), list(im2)
        #
        # with Pool(16) as p:
        #     im1_pred = p.apply_async(np_warp_2D, args=(im2, flow_neg,))
        #     # im1_gt = p.map(np_warp_2D, (im2, flow_gt_neg))
        #     # color_flow_final = p.map(flow_to_color_np, flow)
        #     # color_flow_gt = p.map(flow_to_color_np, flow_gt)
        # p.close()
        # p.join()
        # im1_pred = im1_pred.get()


        im_error = list(im1-im2)
        im_error_gt = list(im1-im1_gt)
        im_error_pred = list(im1-im1_pred)

        results = dict()
        results['img_ref'] = list(im1)
        results['img_mov'] = list(im2)
        results['mov_corr'] = im1_pred
        results['color_flow_pred'] = color_flow_final
        results['color_flow_gt'] = color_flow_gt
        results['err_pred'] = im_error_pred
        results['err_orig'] = im_error
        results['err_gt'] = im_error_gt
        results['flow_pred'] = list(flow_final)
        results['flow_gt'] = list(flow_orig)
        results['loss_pred'] = final_loss
        results['loss_orig'] = final_loss_orig
        results['loss_ang_pred'] = final_loss_angel
        results['loss_ang_orig'] = final_loss_orig_angel

        # results = [np.rot90(i) for i in results]

    return results

def save_test_info(dir, config):
    output_file = os.path.join(dir, 'test_patient_info.txt')
    for slice in config['slice']:
        patient = config['path'].split('/')[-1].split('.')[0]
        f = open(output_file, "a")
        f.write("{},{}\n".format(patient, slice))
        f.close()


def old_save_test_info(dir, config):
    output_file = os.path.join(dir, 'test_patient_info.txt')
    for path in config['test_dir']:
        for frame in config['selected_frames']:
            for slice in config['selected_slices']:
                patient = path.split('/')[-1]
                f = open(output_file, "a")
                f.write("{},{},{}\n".format(patient, frame, slice))
                f.close()


def test_name_map(config):
    if config['u_type'] == 0:
        test_type = 'c'
    elif config['u_type'] == 1:
        test_type = 's'
    elif config['u_type'] == 2:
        test_type = 'r'
    else:
        raise ImportError('wrong test type given')
    return test_type


def save_results(output_dir, results, config, input_cf):
    test_type = test_name_map(input_cf)
    test_name = test_type + '_US' + str(input_cf['US_acc'])

    if config['save_loss']:
        dir_name = 'loss'
        save_dir = os.path.join(output_dir, dir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print("Original Flow Loss: {}".format(results['loss_orig']))
        print("Smoothing Flow Loss: {}".format(results['loss_pred']))
        file_name = test_name + '_EPE_loss.txt'
        output_file_loss = os.path.join(save_dir, file_name)
        f = open(output_file_loss, "a")
        # f.write("{}\n".format(results['loss_orig']))
        f.write("{}\n".format(results['loss_pred']))
        f.close()

        file_name = test_name + '_EAE_loss.txt'
        output_file_loss = os.path.join(save_dir, file_name)
        f = open(output_file_loss, "a")
        # f.write("{}\n".format(results['loss_ang_orig']))
        f.write("{}\n".format(results['loss_ang_pred']))
        f.close()

    patient = input_cf['path'].split('/')[-1].split('.')[0]
    i = 0
    for s in input_cf['slice']:
        file_name = test_type + '_' + patient + '_' + str(input_cf['frame']) + '_' + str(s)

        if config['save_data_npz']:
            dir_name = test_name + '_data_npz'
            save_dir = os.path.join(output_dir, dir_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            output_file_flow = os.path.join(save_dir, file_name)
            np.savez(output_file_flow,
                     img_ref=results['img_ref'][i],
                     flow_gt=results['flow_gt'][i],
                     flow_pred=results['flow_pred'][i])

        if config['save_png']:
            dir_name = test_name + '_png'
            save_dir = os.path.join(output_dir, dir_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            output_file_png = os.path.join(save_dir, file_name)
            save_img(results['img_ref'][i], output_file_png + '_img_ref', 'png')
            save_img(results['img_mov'][i], output_file_png + '_img_mov', 'png')
            save_img(results['mov_corr'][i], output_file_png + '_mov_corr', 'png')
            save_img(results['color_flow_gt'][i], output_file_png + '_flow_gt', 'png')
            save_img(results['color_flow_pred'][i], output_file_png + '_flow_pred', 'png')
        if config['save_pdf']:
            dir_name = test_name + '_pdf'
            save_dir = os.path.join(output_dir, dir_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            output_file_png = os.path.join(save_dir, file_name)
            save_img(results['img_ref'][i], output_file_png+'_img_ref', 'pdf')
        i += 1


def save_img(result, file_path, format='png'):
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.axis('off')
    if len(result.shape) == 2:
        plt.imshow(result, cmap="gray")
    else:
        plt.imshow(result)
    fig.savefig(file_path+'.'+format)
    plt.close()


def show_results(results):
    for i in range(len(results['img_ref'])):
        fig, ax = plt.subplots(3, 3, figsize=(14, 14))
        ax[0][0].imshow(results['img_ref'][i], cmap='gray')
        ax[0][0].set_title('Ref Img')
        ax[0][1].imshow(results['img_mov'][i], cmap='gray')
        ax[0][1].set_title('Moving Img')
        fig.delaxes(ax[0, 2])

        ax[1][0].imshow(results['mov_corr'][i], cmap='gray')
        ax[1][0].set_title('Moving Corrected')
        ax[1][1].imshow(results['color_flow_pred'][i])
        ax[1][1].set_title('Flow Pred')
        ax[1][2].imshow(results['color_flow_gt'][i])
        ax[1][2].set_title('Flow GT')

        ax[2][0].imshow(results['err_pred'][i], cmap='gray')
        ax[2][0].set_title('Warped error')
        ax[2][1].imshow(results['err_orig'][i], cmap='gray')
        ax[2][1].set_title('Original Error')
        ax[2][2].imshow(results['err_gt'][i], cmap='gray')
        ax[2][2].set_title('GT Error')

        plt.show()


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    config = {}
    config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/21_tk',
                          '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/06_la',
                          '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/035']
    config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz']
    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_12_hs.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_035.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_036.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_06_la.npz']

    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_12_hs.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_06_la.npz']

    # 0: constant generated flow, 1: smooth generated flow, 2: cross test without gt, 3: matlab simulated test data
    config['test_types'] = [2]
    config['US_acc'] = [8]
    # config['US_acc'] = list(range(1, 32, 2))
    # config['test_types'] = list(2 * np.ones(len(config['US_acc']), dtype=np.int))

    config['mask_type'] = 'center'
    # config['mask_type'] = 'US'
    config['selected_frames'] = [0]
    config['selected_slices'] = [34]
    config['amplitude'] = 10
    config['network'] = 'flownet'
    config['batch_size'] = 64

    config['save_results'] = True
    config['save_data_npz'] = False
    config['save_loss'] = True
    config['save_pdf'] = False
    config['save_png'] = True

    print("-- evaluating: on {} pairs from {}"
          .format(FLAGS.num, FLAGS.dataset))

    default_config = config_dict()
    dirs = default_config['dirs']

    if FLAGS.dataset == 'kitti':
        data = KITTIData(dirs['data'], development=True)
        data_input = KITTIInput(data, batch_size=1, normalize=False,
                                dims=(384, 1280))

    elif FLAGS.dataset == 'resp_2D':
        kdata = KITTIData(data_dir=dirs['data'], development=True)
        data_input = MRI_Resp_2D(data=kdata,
                                 batch_size=config['batch_size'],
                                 normalize=False,
                                 dims=(256, 256))
        FLAGS.num = 1

    input_cf = dict()
    input_cf['use_given_u'] = True
    input_cf['US'] = True
    input_cf['use_given_US_mask'] = True
    input_cf['cross_test'] = False

    info_file = "/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/slice_info.ods"
    ods = get_data(info_file)
    slice_info = {value[0]: list(range(*[int(j) - 1 for j in value[1].split(',')])) for value in ods["Sheet1"] if
                  len(value) is not 0}
    if 'selected_slices' in config:
        for patient in config['test_dir']:
            name_pat = patient.split('/')[-1].split('.')[0]
            slice_info[name_pat] = config['selected_slices']
    input_cf['slice_info'] = slice_info

    for name in FLAGS.ex.split(','):
        if config['save_results']:
            output_dir = os.path.join("/home/jpa19/PycharmProjects/MA/UnFlow/output/", name+'_test')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        for i, u_type in enumerate(config['test_types']):
            for patient in config['test_dir']:
                for frame in config['selected_frames']:
                    input_cf['path'] = patient

                    name_pat = patient.split('/')[-1].split('.')[0]
                    input_cf['slice'] = slice_info[name_pat]
                    config['selected_slices'] = input_cf['slice']

                    input_cf['frame'] = frame
                    input_cf['u_type'] = u_type
                    # input_cf['use_given_u'] = config['new_u'][i]
                    input_cf['mask_type'] = config['mask_type']
                    input_cf['US_acc'] = config['US_acc'][i]
                    # input_cf['use_given_US_mask'] = config['new_US_mask'][i]
                    if config['save_results']:
                        save_test_info(output_dir, input_cf)

                    results = _evaluate_experiment(name, lambda: data_input.test_flown(config=input_cf), config)
                    # show_results(results)
                    pass
                    save_results(output_dir, results, config, input_cf)
        cal_loss_mean(os.path.join(output_dir, 'loss'))

if __name__ == '__main__':
    tf.app.run()
