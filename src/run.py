#!/usr/bin/python3
import os
import copy
import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.client import device_lib
from pyexcel_ods import get_data

from e2eflow.core.train import Trainer
from e2eflow.experiment import Experiment
from e2eflow.util import convert_input_strings

from e2eflow.kitti.input_resp import KITTIInput, MRI_Resp_2D
from e2eflow.kitti.input_card import MRI_Card_2D
from e2eflow.kitti.data import KITTIData


tf.app.flags.DEFINE_string('ex', 'default',
                           'Name of the experiment.'
                           'If the experiment folder already exists in the log dir, '
                           'training will be continued from the latest checkpoint.')
tf.app.flags.DEFINE_boolean('debug', False,
                            'Enable image summaries and disable checkpoint writing for debugging.')
tf.app.flags.DEFINE_boolean('ow', True,
                            'Overwrites a previous experiment with the same name (if present)'
                            'instead of attempting to continue from its latest checkpoint.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    experiment = Experiment(
        name=FLAGS.ex,
        overwrite=FLAGS.ow)
    dirs = experiment.config['dirs']
    run_config = experiment.config['run']

    gpu_list_param = run_config['gpu_list']
    np.random.seed(1)
    tf.set_random_seed(1)
    random.seed(1)
    os.unsetenv('PYTHONHASHSEED')
    os.environ['PYTHONHASHSEED'] = '1'
    os.unsetenv('TF_CUDNN_DETERMINISTIC')
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    gpu_batch_size = int(run_config['batch_size'] / max(len(gpu_list), 1))
    devices = ['/gpu:' + str(gpu_num) for gpu_num in gpu_list]

    train_dataset = run_config.get('dataset', 'kitti')

    kdata = KITTIData(data_dir=dirs['data'],
                      fast_dir=dirs.get('fast'),
                      stat_log_dir=None,
                      development=run_config['development'])
    einput = KITTIInput(data=kdata,
                        batch_size=1,
                        normalize=False,
                        dims=(384, 1280))

    if train_dataset == 'resp_2D':
        info_file = "/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/slice_info_resp.ods"
        ods = get_data(info_file)
        slice_info = {value[0]: list(range(*[int(j) - 1 for j in value[1].split(',')])) for value in ods["Sheet1"] if
                      len(value) is not 0}

        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_resp_2D'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = MRI_Resp_2D(data=kdata,
                              batch_size=ftconfig.get('batch_size'),
                              normalize=False,
                              dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
            lambda: ftinput.input_train_data(img_dirs=['resp/new_data/npz/train'],
                                             slice_info=slice_info,
                                             params=ftconfig,
                                             case='train'),
            lambda: ftinput.input_train_data(img_dirs=['resp/new_data/npz/test/'],
                                             slice_info=slice_info,
                                             params=ftconfig,
                                             case='validation'),
            supervised=True,
            params=ftconfig,
            normalization=ftinput.get_normalization(),
            train_summaries_dir=experiment.train_dir,
            eval_summaries_dir=experiment.eval_dir,
            experiment=FLAGS.ex,
            ckpt_dir=experiment.save_dir,
            debug=FLAGS.debug,
            interactive_plot=run_config.get('interactive_plot'),
            devices=devices,
            LAP_layer=ftconfig.get('lap_layer'))
        tr.run(0, ftiters)

    elif train_dataset == 'card_2D':
        info_file = "/home/jpa19/PycharmProjects/MA/UnFlow/data/card/slice_info_card.ods"
        ods = get_data(info_file)
        slice_info = {value[0]: list(range(*[int(j) - 1 if i == 0 else int(j)
                      for i, j in enumerate(value[1].split(','))])) for value in ods["Sheet1"] if len(value) is not 0}

        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_card_2D'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = MRI_Card_2D(data=kdata,
                              batch_size=ftconfig.get('batch_size'),
                              normalize=False,
                              dims=(ftconfig['desired_height'], ftconfig['desired_width']))
        tr = Trainer(
            lambda: ftinput.input_train_data(img_dirs=['card/npz/train'],
                                             slice_info=slice_info,
                                             params=ftconfig,
                                             case='train'),
            lambda: ftinput.input_train_data(img_dirs=['card/npz/test/'],
                                             slice_info=slice_info,
                                             params=ftconfig,
                                             case='validation'),
            supervised=True,
            params=ftconfig,
            normalization=ftinput.get_normalization(),
            train_summaries_dir=experiment.train_dir,
            eval_summaries_dir=experiment.eval_dir,
            experiment=FLAGS.ex,
            ckpt_dir=experiment.save_dir,
            debug=FLAGS.debug,
            interactive_plot=run_config.get('interactive_plot'),
            devices=devices,
            LAP_layer=ftconfig.get('lap_layer'))
        tr.run(0, ftiters)
    elif train_dataset == 'kitti_ft':
        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_kitti_ft'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = KITTIInput(data=kdata,
                             batch_size=gpu_batch_size,
                             normalize=False,
                             dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
              lambda shift: ftinput.input_train_gt(40),
              lambda: einput.input_train_2015(40),
              supervised=True,
              params=ftconfig,
              normalization=ftinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, ftiters)
    elif train_dataset == 'mri_resp_3D':
        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_resp_3D'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = MRI_Resp_3D(data=kdata,
                              batch_size=gpu_batch_size,
                              normalize=False,
                              dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
            lambda shift: ftinput.input_train_gt(40),
            lambda: einput.input_train_2015(40),
            supervised=True,
            params=ftconfig,
            normalization=ftinput.get_normalization(),
            train_summaries_dir=experiment.train_dir,
            eval_summaries_dir=experiment.eval_dir,
            experiment=FLAGS.ex,
            ckpt_dir=experiment.save_dir,
            debug=FLAGS.debug,
            interactive_plot=run_config.get('interactive_plot'),
            devices=devices)
        tr.run(0, ftiters)

    else:
      raise ValueError(
          "Invalid dataset. Dataset must be one of "
          "{synthia, kitti, kitti_ft, cityscapes, chairs}")

    if not FLAGS.debug:
        experiment.conclude()


if __name__ == '__main__':
    tf.app.run()
