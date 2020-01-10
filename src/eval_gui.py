import os
import sys
import shutil
import PIL
import tensorflow as tf
import numpy as np
import png
import matplotlib.pyplot as plt
import pylab

from e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct
from e2eflow.core.flow_util import flow_error_image
from e2eflow.util import config_dict
from e2eflow.core.image_warp import image_warp
from e2eflow.kitti.input import KITTIInput
from e2eflow.kitti.data import KITTIData
from e2eflow.kitti.input import MRI_Resp_2D
from e2eflow.core.supervised import supervised_loss
from e2eflow.core.input import resize_input, resize_output_crop, resize_output, resize_output_flow
from e2eflow.core.train import restore_networks
from e2eflow.ops import forward_warp
from e2eflow.gui import display
from e2eflow.core.losses import DISOCC_THRESH, occlusion, create_outgoing_mask
from e2eflow.util import convert_input_strings
from e2eflow.kitti.input import np_warp_2D

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


def write_rgb_png(z, path, bitdepth=8):
    z = z[0, :, :, :]
    with open(path, 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=bitdepth)
        z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
        writer.write(f, z2list)


def flow_to_int16(flow):
    num_batch, h, w, _ = tf.unstack(tf.shape(flow))
    u, v = tf.unstack(flow, num=2, axis=3)
    r = tf.cast(tf.maximum(0.0, tf.minimum(u * 64.0 + 32768.0, 65535.0)), tf.uint16)
    g = tf.cast(tf.maximum(0.0, tf.minimum(v * 64.0 + 32768.0, 65535.0)), tf.uint16)
    b = tf.ones([num_batch, h, w], tf.uint16)
    return tf.stack([r, g, b], axis=3)


def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    flow = flow[0, :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()


def _evaluate_experiment(name, data_input, test_path, selected_frames, selected_slices, cross_test=False, LAP=False):
    normalize_fn = data_input._normalize_image
    resized_h = data_input.dims[0]
    resized_w = data_input.dims[1]

    current_config = config_dict('../config.ini')
    exp_dir = os.path.join(current_config['dirs']['log'], 'ex', name)
    config_path = os.path.join(exp_dir, 'config.ini')
    if not os.path.isfile(config_path):
        config_path = '../config.ini'
    if not os.path.isdir(exp_dir) or not tf.train.get_checkpoint_state(exp_dir):
        exp_dir = os.path.join(current_config['dirs']['checkpoints'], name)
    config = config_dict(config_path)
    params = config['train']
    convert_input_strings(params, config_dict('../config.ini')['dirs'])
    dataset_params_name = 'train_' + FLAGS.dataset
    if dataset_params_name in config:
        params.update(config[dataset_params_name])
    ckpt = tf.train.get_checkpoint_state(exp_dir)
    if not ckpt:
        raise RuntimeError("Error: experiment must contain a checkpoint")
    ckpt_path = exp_dir + "/" + os.path.basename(ckpt.model_checkpoint_path)

    with tf.Graph().as_default(): #, tf.device('gpu:' + FLAGS.gpu):
        input_fn = data_input.input_test_data(test_path, selected_frames, selected_slices, cross_test=cross_test)
        im1, im2, flow_gt = input_fn
        height = tf.shape(im1)[1]
        width = tf.shape(im1)[2]
        im1 = im1[:, 0, :, :]
        im2 = im2[:, 0, :, :]
        flow_gt = flow_gt[:, 0, :, :, :]
        # mask = mask[:, 0, :, :]
        im1 = im1[..., tf.newaxis]
        im2 = im2[..., tf.newaxis]
        # mask = mask[..., tf.newaxis]

        loss, flow = supervised_loss(
                     input_fn,
                     normalization=data_input.get_normalization(),
                     augment=False,
                     params=params,
                     LAP=LAP)

        # im1 = resize_output(im1, height, width, 3)
        # im2 = resize_output(im2, height, width, 3)
        # flow = resize_output_flow(flow, height, width, 2)

        flow_fw_int16 = flow_to_int16(flow)

        im1_pred = image_warp(im2, -flow)
        im1_diff = tf.abs(im1 - im1_pred)
        ori_diff = tf.abs(im1 - im2)
        flow_diff = tf.abs(flow - flow_gt)

        # flow_gt = resize_output_crop(flow_gt, height, width, 2)
        # mask = resize_output_crop(mask, height, width, 1)

        image_slots = [(im1 / 255, 'first image'),
                       (im2 / 255, 'second image'),
                       (im1_pred, 'warped second image'),
                       (ori_diff, 'original error'),
                       (im1_diff, 'warping error'),
                       (flow_to_color(flow), 'flow'),
                       (flow_to_color(flow_gt), 'gt_flow')
                       ]

        # list of (scalar_op, title)
        scalar_slots = [(flow_error_avg(flow_gt, flow, tf.ones(tf.shape(flow))), 'EPE_all')]


        num_ims = len(image_slots)
        image_ops = [t[0] for t in image_slots]
        scalar_ops = [t[0] for t in scalar_slots]
        image_names = [t[1] for t in image_slots]
        scalar_names = [t[1] for t in scalar_slots]
        all_ops = image_ops + scalar_ops

        image_lists = []
        averages = np.zeros(len(scalar_ops))
        sess_config = tf.ConfigProto(allow_soft_placement=True)

        exp_out_dir = os.path.join('../out', name)
        if FLAGS.output_visual or FLAGS.output_benchmark:
            if os.path.isdir(exp_out_dir):
                shutil.rmtree(exp_out_dir)
            os.makedirs(exp_out_dir)
            shutil.copyfile(config_path, os.path.join(exp_out_dir, 'config.ini'))

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restore_networks(sess, params, ckpt, ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,
                                                   coord=coord)

            # TODO adjust for batch_size > 1 (also need to change image_lists appending)
            max_iter = FLAGS.num if FLAGS.num > 0 else None

            try:
                num_iters = 0
                while not coord.should_stop() and (max_iter is None or num_iters != max_iter):
                    all_results = sess.run([flow, flow_fw_int16, loss] + all_ops)
                    flow_gt = sess.run(flow_gt)
                    flow_fw_res, flow_fw_int16_res = all_results[:2]
                    loss_res = all_results[2]
                    all_results = all_results[3:]
                    image_results = all_results[:num_ims]
                    flow_diff = flow_gt - flow_fw_res  # JP
                    scalar_results = all_results[num_ims:]
                    iterstr = str(num_iters).zfill(6)

                    if FLAGS.output_visual:
                        path_im1 = os.path.join(exp_out_dir, iterstr + '_im1.png')
                        path_im2 = os.path.join(exp_out_dir, iterstr + '_im2.png')
                        path_warp_im2 = os.path.join(exp_out_dir, iterstr + '_im2_wrapped.png')
                        path_warp_err = os.path.join(exp_out_dir, iterstr + '_warp_err.png')
                        path_flow = os.path.join(exp_out_dir, iterstr + '_flow.png')
                        path_flow_gt = os.path.join(exp_out_dir, iterstr + '_flow_gt.png')
                        write_rgb_png(image_results[0] * 255, path_im1)
                        write_rgb_png(image_results[1] * 255, path_im2)
                        write_rgb_png(image_results[2] * 255, path_warp_im2)
                        write_rgb_png(image_results[3] * 255, path_warp_err)
                        write_rgb_png(image_results[4] * 255, path_flow)
                        write_rgb_png(image_results[5] * 255, path_flow_gt)

                    if FLAGS.output_benchmark:
                        path_fw = os.path.join(exp_out_dir, iterstr)
                        if FLAGS.output_png:
                            write_rgb_png(flow_fw_int16_res, path_fw  + '_10.png', bitdepth=16)
                        else:
                            write_flo(flow_fw_res, path_fw + '_10.flo')
                    if num_iters < FLAGS.num_vis:
                        image_lists.append(image_results)
                    averages += scalar_results
                    if num_iters > 0:
                        sys.stdout.write('\r')
                    num_iters += 1
                    sys.stdout.write("-- evaluating '{}': {}/{}"
                                     .format(name, num_iters, max_iter))
                    sys.stdout.flush()
                    print()
                    print('charbonnier_loss = ' + str(loss_res))
            except tf.errors.OutOfRangeError:
                pass

            averages /= num_iters

            coord.request_stop()
            coord.join(threads)

    for t, avg in zip(scalar_slots, averages):
        _, scalar_name = t
        print("({}) {} = {}".format(name, scalar_name, avg))

    return image_lists, image_names

# def show_results(result, save_path=None):
#     f = plt.figure()
#     f.add_subplot(3, 2, 1)
#     plt.imshow(result[0][0][0][0, :, :, 0], cmap='gray')
#     f.add_subplot(3, 2, 2)
#     plt.imshow(result[0][0][1][0, :, :, 0], cmap='gray')

def show_results(result, save_path=None):

    for num in range(np.shape(result[0][0][0])[0]):
        fig, ax = plt.subplots(2, 4, figsize=(15, 8))
        ax[0][0].imshow(result[0][0][0][num, :, :, 0], cmap='gray')  # ref
        ax[0][0].set_title('Orignal Img')
        ax[0][1].imshow(result[0][0][1][num, :, :, 0], cmap='gray')  # mov
        ax[0][1].set_title('Moving Img')
        ax[0][2].imshow(result[0][0][2][num, :, :, 0], cmap='gray')  # warped
        ax[0][2].set_title('Moving Corrected')
        fig.delaxes(ax[0, 3])
        ax[1][0].imshow(result[0][0][3][num, :, :, 0], cmap='gray')
        ax[1][0].set_title('Original Error')

        ax[1][1].imshow(result[0][0][4][num, :, :, 0], cmap='gray')
        ax[1][1].set_title('Warped Error')

        ax[1][2].imshow(result[0][0][5][num, ...])
        # ax[1][1].imshow(result[0][0][4][0, :, :, 0] * 255, cmap='jet', vmin=0, vmax=3.0)
        ax[1][2].set_title('Predicted Flow')

        ax[1][3].imshow(result[0][0][6][num, ...])
        # ax[1][2].imshow(result[0][0][5][0, :, :, 0] * 255, cmap='jet', vmin=0, vmax=3.0)
        ax[1][3].set_title('GT Flow')
        plt.show()
        if save_path:
            plt.savefig(os.path.join(save_path, '.' + 'pdf'),
                    format='pdf')


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    test_dir = '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/'
    test_data = '001/Ph4_Tol100_t000_Ext00_EspOff_closest_recon.mat'
    selected_frames = [0, 3]
    #selected_slices = list(range(15, 55))
    selected_slices = [35]
    LAP = False
    cross_test = False

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
                                 batch_size=4,
                                 normalize=False,
                                 dims=(256, 256))
        FLAGS.num = 1
    # input_fn = getattr(data_input, 'input_' + FLAGS.variant)
    test_path = os.path.join(test_dir, test_data)
    results = []
    for name in FLAGS.ex.split(','):
        result, image_names = _evaluate_experiment(name,
                                                   data_input,
                                                   test_path,
                                                   selected_frames,
                                                   selected_slices,
                                                   cross_test=cross_test,
                                                   LAP=LAP)
        results.append(result)

    # display(results, image_names)
    show_results(results)

    pass


if __name__ == '__main__':
    tf.app.run()
