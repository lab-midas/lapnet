import os
import sys
import shutil
import PIL
import tensorflow as tf
import numpy as np
from scipy import signal
import pylab as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import operator
from scipy.interpolate import griddata
from pyexcel_ods import get_data
import time

from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct, flow_to_color_np
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.data import KITTIData
from e2eflow.kitti.input_resp import MRI_Resp_2D, np_warp_2D, KITTIInput
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


def central_crop(img, bounding):
    """
    central crop for 2D/3D arrays
    # alternative code:
    cutting_part = int((crop_size - 1)/2)
    flow_gt_cut = flow_gt[cutting_part:(np.shape(flow_gt)[0] - cutting_part - 1),
    cutting_part:(np.shape(flow_gt)[0] - cutting_part - 1), :]

    :param img:
    :param bounding:
    :return:
    """
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


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
    crop_stride = config['crop_stride']
    smooth_wind_size = config['smooth_wind_size']
    height = params['height']
    width = params['width']
    height = 192
    width = 156
    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        test_batch, im1, im2, flow_orig, pos = data()

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
            threads = tf.train.start_queue_runners(sess=sess,
                                                   coord=coord)

            methods = ['smmoth', 'average', 'interp']
            method = methods[1]
            if method is 'smooth':
                flow_raw = np.zeros((height, width, 2), dtype=np.float32)
                # flow_raw = np.zeros((int(np.sqrt(np.shape(pos)[0])), int(np.sqrt(np.shape(pos)[0])), 2), dtype=np.float32)
                for i in range(int(np.floor(len(pos)/batch_size)) + 1):
                    flow_pixel, loss_pixel = sess.run([flow, loss])
                    local_pos = pos[batch_size*i:batch_size*i+batch_size, :]
                    try:
                        flow_raw[local_pos[:, 0], local_pos[:, 1], :] = flow_pixel
                    except Exception:  # for the last patches
                        last_batch_size = len(local_pos)
                        flow_raw[local_pos[:, 0], local_pos[:, 1], :] = flow_pixel[:last_batch_size, :]

                if crop_stride is not 1:
                   pass

                if smooth_wind_size is not None:
                    smooth_wind = 1/smooth_wind_size/smooth_wind_size * \
                                  np.ones((smooth_wind_size, smooth_wind_size), dtype=np.float32)
                    flow_final_x = signal.convolve2d(flow_raw[..., 0], smooth_wind, mode='same')
                    flow_final_y = signal.convolve2d(flow_raw[..., 1], smooth_wind, mode='same')
                    flow_final = np.stack((flow_final_x, flow_final_y), axis=-1)
                else:
                    flow_final = np.copy(flow_raw)
            elif method is 'interp':
                flow_raw = np.zeros((np.shape(pos)[0], np.shape(pos)[1]), dtype=np.float32)
                time_start = time.time()
                for i in range(int(np.floor(len(pos)/batch_size)) + 1):
                    flow_pixel, loss_pixel = sess.run([flow, loss])
                    if batch_size*i+batch_size <= len(pos):
                        flow_raw[batch_size*i:batch_size*i+batch_size, :] = flow_pixel
                    else:
                        flow_raw[batch_size*i:batch_size*i+batch_size, :] = flow_pixel[:len(pos)-batch_size*i, :]

                grid_x, grid_y = np.mgrid[0:256:1, 0:256:1]
                flow_final = griddata(pos, flow_raw, (grid_x, grid_y), method='linear', fill_value=0)
                time_end = time.time()
                print('time cost: {}s'.format(time_end - time_start))
            elif method is 'average':
                flow_raw = np.zeros((height, width, 2), dtype=np.float32)
                time_start = time.time()
                smooth_radius = int((smooth_wind_size - 1) / 2)
                counter_mask = np.zeros((height, width, 2), dtype=np.float32)
                for i in range(int(np.floor(len(pos)/batch_size)) + 1):
                    flow_pixel, loss_pixel = sess.run([flow, loss])
                    local_pos = pos[batch_size*i:batch_size*i+batch_size, :]
                    for j in range(len(local_pos)):
                        lower_bound_x = max(0, local_pos[j, 0]-smooth_radius)
                        upper_bound_x = min(height, local_pos[j, 0]+smooth_radius+1)
                        lower_bound_y = max(0, local_pos[j, 1]-smooth_radius)
                        upper_bound_y = min(width, local_pos[j, 1]+smooth_radius+1)
                        flow_raw[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y, :] += flow_pixel[j, :]
                        counter_mask[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y, :] += 1
                flow_final = flow_raw/counter_mask
                time_end = time.time()
                print('time cost: {}s'.format(time_end - time_start))

            coord.request_stop()
            coord.join(threads)


            flow_gt = np.squeeze(flow_orig)
            im1 = np.squeeze(im1)
            im2 = np.squeeze(im2)

            cut_size = (height, width)
            flow_gt_cut = central_crop(flow_gt, cut_size)
            im1_cut = central_crop(im1, cut_size)
            im2_cut = central_crop(im2, cut_size)

            im1_pred = np_warp_2D(im2_cut, -flow_final)

            im_error = im1_cut - im2_cut
            im_error_pred = im1_cut - im1_pred

            # warped error of GT
            im1_gt = np_warp_2D(im2_cut, -flow_gt_cut)
            im1_error_gt = im1_cut - im1_gt

            u_GT = (flow_gt_cut[..., 0], flow_gt_cut[..., 1])  # tuple
            u_est = (flow_final[..., 0], flow_final[..., 1])  # tuple
            OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
            error_data_pred = warp_assessment3D(u_GT, u_est, OF_index)

            size_mtx = np.shape(flow_gt_cut[..., 0])
            u_GT = (np.zeros(size_mtx, dtype=np.float32), np.zeros(size_mtx, dtype=np.float32))  # tuple
            u_est = (flow_gt_cut[..., 0], flow_gt_cut[..., 1])  # tuple
            OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
            error_data_gt = warp_assessment3D(u_GT, u_est, OF_index)

            final_loss_orig = error_data_gt['Abs_Error_mean']
            final_loss = error_data_pred['Abs_Error_mean']
            final_loss_orig_angel = error_data_gt['Angle_Error_Mean']
            final_loss_angel = error_data_pred['Angle_Error_Mean']

            # error_orig = flow_gt_cut
            # error_final = flow_final - flow_gt_cut
            # # error_raw = flow_raw - flow_gt_cut
            #
            # final_loss_orig = np.mean(np.sqrt(np.sum(np.square(error_orig), 2)))
            # # final_loss_orig = np.mean(np.square(error_orig))
            # final_loss = np.mean(np.sqrt(np.sum(np.square(error_final), 2)))
            # # final_loss = np.mean(np.square(error_final))

            # final_loss_raw = np.mean(np.square(error_raw))
            # print("Raw Flow Loss: {}".format(final_loss_raw))

            # flow_raw = flow_to_color_np(flow_raw, convert_to_bgr=False)
            color_flow_final = flow_to_color_np(flow_final, convert_to_bgr=False)
            color_flow_gt = flow_to_color_np(flow_gt_cut, convert_to_bgr=False)
            # flow_error = flow_to_color_np(error_final, convert_to_bgr=False)

            compare_the_flow = False
            if compare_the_flow:
                # compare the raw and smoothed flow
                fig, ax = plt.subplots(1, 3, figsize=(8, 4))
                ax[0].imshow(flow_gt)  # ref
                ax[0].set_title('Flow GT')
                ax[1].imshow(flow_raw)  # mov
                ax[1].set_title('Flow Pred Raw')
                ax[2].imshow(flow_final)
                ax[2].set_title('Flow Pred Smooth')
                plt.show()

            results = dict()
            results['img_ref'] = im1_cut
            results['img_mov'] = im2_cut
            results['mov_corr'] = im1_pred
            results['color_flow_pred'] = color_flow_final
            results['color_flow_gt'] = color_flow_gt
            results['err_pred'] = im_error_pred
            results['err_orig'] = im_error
            results['err_gt'] = im1_error_gt
            results['flow_pred'] = flow_final
            results['flow_gt'] = flow_gt_cut
            results['loss_pred'] = final_loss
            results['loss_orig'] = final_loss_orig
            results['loss_ang_pred'] = final_loss_angel
            results['loss_ang_orig'] = final_loss_orig_angel

            # results = [np.rot90(i) for i in results]

    return results


def save_test_info(dir, config):
    output_file = os.path.join(dir, 'test_patient_info.txt')
    patient = config['path'].split('/')[-1].split('.')[0]
    f = open(output_file, "a")
    f.write("{},{}\n".format(patient, config['slice']))
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
    file_name = test_type + '_' + patient + '_' + str(input_cf['frame']) + '_' + str(input_cf['slice'])

    if config['save_data_npz']:
        dir_name = test_name + '_data_npz'
        save_dir = os.path.join(output_dir, dir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        output_file_flow = os.path.join(save_dir, file_name)
        np.savez(output_file_flow,
                 img_ref=results['img_ref'],
                 flow_gt=results['flow_gt'],
                 flow_pred=results['flow_pred'])

    if config['save_png']:
        dir_name = test_name + '_png'
        save_dir = os.path.join(output_dir, dir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        output_file_png = os.path.join(save_dir, file_name)
        save_img(results['img_ref'], output_file_png + '_img_ref', 'png')
        save_img(results['img_mov'], output_file_png + '_img_mov', 'png')
        save_img(results['mov_corr'], output_file_png + '_mov_corr', 'png')
        save_img(results['color_flow_gt'], output_file_png + '_flow_gt', 'png')
        save_img(results['color_flow_pred'], output_file_png + '_flow_pred', 'png')
    if config['save_pdf']:
        dir_name = test_name + '_pdf'
        save_dir = os.path.join(output_dir, dir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        output_file_png = os.path.join(save_dir, file_name)
        save_img(results['img_ref'], output_file_png+'_img_ref', 'pdf')


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
    fig, ax = plt.subplots(3, 3, figsize=(14, 14))
    plt.axis('off')
    ax[0][0].imshow(results['img_ref'], cmap='gray')
    ax[0][0].set_title('Ref Img')
    ax[0][0].axis('off')
    ax[0][1].imshow(results['img_mov'], cmap='gray')
    ax[0][1].set_title('Moving Img')
    ax[0][1].axis('off')
    fig.delaxes(ax[0, 2])

    ax[1][0].imshow(results['mov_corr'], cmap='gray')
    ax[1][0].set_title('Moving Corrected')
    ax[1][0].axis('off')
    ax[1][1].imshow(results['color_flow_pred'])
    ax[1][1].set_title('Flow Pred')
    ax[1][1].axis('off')
    ax[1][2].imshow(results['color_flow_gt'])
    ax[1][2].set_title('Flow GT')
    ax[1][2].axis('off')

    ax[2][0].imshow(results['err_pred'], cmap='gray')
    ax[2][0].set_title('Warped error')
    ax[2][0].axis('off')
    ax[2][1].imshow(results['err_orig'], cmap='gray')
    ax[2][1].set_title('Original Error')
    ax[2][1].axis('off')
    ax[2][2].imshow(results['err_gt'], cmap='gray')
    ax[2][2].set_title('GT Error')
    ax[2][2].axis('off')

    plt.show()
    #plt.savefig("test.png", bbox_inches='tight')


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    config = dict()
    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/21_tk',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/06_la',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/035']

    config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz']
    config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/card/npz/test/Pat1.npz']
    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_12_hs.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_035.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_036.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_06_la.npz']

    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_12_hs.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/patient_004.npz',
    #                       '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/test/volunteer_06_la.npz']

    # 0: constant generated flow, 1: smooth generated flow, 2: matlab simulated test data 3: simulated_x smooth 4: cross test without gt
    config['test_types'] = [2]
    config['US_acc'] = [1]
    # config['US_acc'] = list(range(1, 32, 2))
    # config['test_types'] = list(2*np.ones(len(config['US_acc']), dtype=np.int))


    config['mask_type'] = 'center'
    # config['mask_type'] = 'US'

    config['selected_frames'] = [0]
    config['selected_slices'] = [11]
    config['amplitude'] = 10
    config['network'] = 'ftflownet'
    config['batch_size'] = 64
    config['smooth_wind_size'] = 17  # None for no smoothing
    config['crop_stride'] = 2
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
                                 dims=(192, 174))
        FLAGS.num = 1


    # for name in FLAGS.ex.split(','):
    #     results = _evaluate_experiment(name, lambda: data_input.input_patch_test_data(config=config))
    #     show_results(results)
    #     # results.append(result)

    input_cf = dict()
    input_cf['use_given_u'] = True
    input_cf['US'] = True
    input_cf['use_given_US_mask'] = True
    input_cf['padding'] = True
    input_cf['crop_size'] = 33
    input_cf['crop_stride'] = config['crop_stride']
    input_cf['cross_test'] = False

    info_file = "/home/jpa19/PycharmProjects/MA/UnFlow/data/card/slice_info_card.ods"
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
            output_dir = os.path.join("/home/jpa19/PycharmProjects/MA/UnFlow/output/", name+'_card1')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        for i, u_type in enumerate(config['test_types']):
            for patient in config['test_dir']:
                for frame in config['selected_frames']:
                    name_pat = patient.split('/')[-1].split('.')[0]
                    config['selected_slices'] = slice_info[name_pat]
                    for slice in config['selected_slices']:

                        input_cf['path'] = patient
                        input_cf['frame'] = frame
                        input_cf['slice'] = slice
                        input_cf['u_type'] = u_type
                        input_cf['mask_type'] = config['mask_type']
                        # input_cf['use_given_u'] = config['new_u'][i]
                        input_cf['US_acc'] = config['US_acc'][i]
                        # input_cf['use_given_US_mask'] = config['new_US_mask'][i]
                        if config['save_results']:
                            save_test_info(output_dir, input_cf)

                        results = _evaluate_experiment(name, lambda: data_input.test_2D_slice(config=input_cf), config)
                        # show_results(results)

                        if config['save_results']:
                            save_results(output_dir, results, config, input_cf)
        cal_loss_mean(os.path.join(output_dir, 'loss'))


if __name__ == '__main__':
    tf.app.run()
