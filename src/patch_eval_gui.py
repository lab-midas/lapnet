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


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    config = {}

    config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/001']
    # config['test_dir'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/patient/030']
    config['test_dir_matlab_simulated'] = ['/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/test_data/matlab_simulated_data']
    # 0: constant generated flow, 1: smooth generated flow, 2: cross test without gt, 3: matlab simulated test data
    config['test_types'] = [0]
    config['selected_frames'] = [0]
    # config['selected_slices'] = list(range(15, 55))
    config['selected_slices'] = [40]
    config['amplitude'] = 10
    config['crop'] = True
    config['test_in_kspace'] = True
    config['cross_test'] = False
    config['batch_size'] = 64

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
    # input_fn = getattr(data_input, 'input_' + FLAGS.variant)
    # test_path = os.path.join(test_dir, test_data)
    results = []

    for name in FLAGS.ex.split(','):

        # current_config = config_dict('../config.ini')
        # exp_dir = os.path.join(current_config['dirs']['log'], 'ex', name)
        # config_path = os.path.join(exp_dir, 'config.ini')
        # if not os.path.isfile(config_path):
        #     config_path = '../config.ini'
        # config = config_dict(config_path)
        # params = config['train']
        # convert_input_strings(params, config_dict('../config.ini')['dirs'])
        # dataset_params_name = 'train_' + FLAGS.dataset
        # if dataset_params_name in config:
        #     params.update(config[dataset_params_name])

        result, image_names = _evaluate_experiment(name, lambda: data_input.input_test_data(config=config))
        results.append(result)

    # display(results, image_names)
    show_results(results)

    pass


if __name__ == '__main__':
    tf.app.run()
