import tensorflow as tf
import yaml
import os

from core.lapnet import buildLAPNet_model_2D, buildLAPNet_model_3D, buildLAPNet_model_2D_old
from core.train import train
from resp_and_card.training_data_2D import save_2D_LAPNet_data_as_npz
from resp_and_card.training_data_3D import save_3D_LAPNet_data_as_npz
from test.eval_lapnet import eval_tapering, eval_cropping
from resp_and_card.test_data_2D import create_2D_test_dataset

from tensorflow.keras import backend as K

# ===============================================================================
# read config file
# ===============================================================================
# read yaml file
with open("config.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
general_setup = data_loaded['Setup']
mode_run = general_setup['mode_run']
if mode_run == 'train':
    data_setup = data_loaded['Train']['training_data']
    experiment_setup = data_loaded['Train']['Experiment']
elif mode_run == 'test':
    data_setup = data_loaded['Test']['test_data']
    experiment_setup = data_loaded['Test']['Evaluate']

# ===============================================================================
# GPU setting
# ===============================================================================
gpu_num = general_setup['gpu_list']
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num  # gpu 3
if gpu_num is not "-1":
    num_Gb = general_setup['gpu_num_gb']
    memory = num_Gb * 1024
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)

# ===============================================================================
# running mode
# ===============================================================================
slicing_mode = general_setup['slicing_mode']
create_data = data_setup['create_data']
dimensionality = general_setup['dimensionality']
architecture_version = general_setup['architecture_version']

# ===============================================================================
# Create training data
# ===============================================================================
if slicing_mode == 'tapering':
    if create_data and mode_run == 'train':
        if dimensionality == '2D':
            save_2D_LAPNet_data_as_npz(data_setup)
        if dimensionality == '3D':
            save_3D_LAPNet_data_as_npz(data_setup)

# ===============================================================================
# Create test data
# ===============================================================================
if slicing_mode == 'tapering':
    if create_data and mode_run == 'test':
        if dimensionality == '2D':
            create_2D_test_dataset(data_setup)
        if dimensionality == '3D':
            pass
            # .. ToDo: function for 3D test data creation

# ===============================================================================
# Build the model
# ===============================================================================
if dimensionality == '2D':
    if architecture_version == 0:
        ModelResp = buildLAPNet_model_2D_old(33)
        print(ModelResp.summary())
    else:
        ModelResp = buildLAPNet_model_2D()
        print(ModelResp.summary())
if dimensionality == '3D':
    ModelResp = buildLAPNet_model_3D()
    print(ModelResp.summary())

# ===============================================================================
# Train the Model
# ===============================================================================
if mode_run == 'train':
    train(ModelResp, general_setup, experiment_setup)

# ===============================================================================
# Test the Model
# ===============================================================================
if mode_run == 'test':
    if slicing_mode == 'tapering':
        res = eval_tapering(ModelResp, experiment_setup, dimensionality)

    if slicing_mode == 'cropping':
        res = eval_cropping(ModelResp, experiment_setup)

# ===============================================================================
# Clear session
# ===============================================================================
K.clear_session()
