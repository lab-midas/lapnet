#!/usr/bin/python3
import os
import copy
import json
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

from e2eflow.core.train import Trainer
from e2eflow.experiment import Experiment
from e2eflow.util import convert_input_strings

from e2eflow.kitti.input import KITTIInput, MRI_Resp_3D, MRI_Resp_2D
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

    if train_dataset == 'mri_resp_3D':
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
    elif train_dataset == 'resp_2D':
        np.random.seed(0)
        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_resp_2D'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = MRI_Resp_2D(data=kdata,
                           batch_size=gpu_batch_size,
                           normalize=False,
                           dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
            lambda: ftinput.input_train_data(img_dirs=['resp/patient', 'resp/volunteer'],
                                             img_dirs_real_simulated=['resp/matlab_simulated_data'],
                                             data_per_interval=ftconfig.get('data_per_interval'),
                                             selected_frames=[0, 3],
                                             selected_slices=list(range(20, 60)),
                                             augment_type_percent=ftconfig.get('augment_type_percent'),
                                             amplitude=ftconfig.get('flow_amplitude'),
                                             train_in_kspace=ftconfig.get('k_space'),
                                             crop=ftconfig.get('crop')),
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
            devices=devices,
            LAP_layer=ftconfig.get('lap_layer'))
        tr.run(0, ftiters)
    elif train_dataset == 'card_2D':
        pass  # todo
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

    else:
      raise ValueError(
          "Invalid dataset. Dataset must be one of "
          "{synthia, kitti, kitti_ft, cityscapes, chairs}")

    if not FLAGS.debug:
        experiment.conclude()


if __name__ == '__main__':
    tf.app.run()
