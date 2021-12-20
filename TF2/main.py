import yaml
from models import *
from train import train_supervised, train_unsupervised
from preprocess.training_data_2D import save_2D_LAPNet_data_as_npz
from preprocess.training_data_3D import save_3D_LAPNet_data_as_npz
from evaluate import eval_tapering, eval_cropping, eval_img
from preprocess.test_data_2D import create_2D_test_dataset
from tensorflow.keras import backend as K
import tensorflow as tf

# ===============================================================================
# read config file
# ===============================================================================
# read yaml file
with open("/home/studghoul1/lapnet/lapnet/TF2/config.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
general_setup = data_loaded['Setup']
mode_run = general_setup['mode_run']
supervised = general_setup['supervised']
if mode_run == 'train_supervised' or mode_run == 'train_unsupervised':
    data_setup = data_loaded['Train']['training_data']
    experiment_setup = data_loaded['Train']['Experiment']
else:
    experiment_setup = data_loaded['Test']['Evaluate']

# ===============================================================================
# GPU setting
# ===============================================================================
# Set up the GPU parameters
gpu_num = general_setup['gpu_list']
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
if gpu_num != "-1":
    num_Gb = general_setup['gpu_num_gb']
    memory = 20480
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
# Fetch general mode settings
slicing_mode = general_setup['slicing_mode']
create_data = False #general_setup['create_data']
dimensionality = general_setup['dimensionality']
architecture_version = general_setup['architecture_version']

# ===============================================================================
# Create training data
# ===============================================================================
if slicing_mode == 'tapering':
    if create_data and mode_run == 'train_supervised':
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
if supervised:
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
else:
    ModelResp = buildLAPNet_model_2D_unsupervised()
    print(ModelResp.summary())
    pass

# ===============================================================================
# Train the Model
# ===============================================================================
if mode_run == 'train_supervised':
    train_supervised(ModelResp, general_setup, experiment_setup)
if mode_run == 'train_unsupervised':
    train_unsupervised(ModelResp, general_setup, experiment_setup)
# ===============================================================================
# Test the Model
# ===============================================================================
if mode_run == 'test':
    if slicing_mode == 'tapering':
        res = eval_tapering(ModelResp, experiment_setup, dimensionality, supervised)
        eval_img(res)

    if slicing_mode == 'cropping':
        res = eval_cropping(ModelResp, experiment_setup)

# ===============================================================================
# Clear session
# ===============================================================================
K.clear_session()
