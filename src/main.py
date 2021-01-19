from pyexcel_ods import get_data
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from e2eflow.resp_and_card.input_resp import DataGenerator_Resp_cropping_2D, DataGenerator_Resp_tapering_2D
from e2eflow.core.flownet import buildLAPNet_model, End_Point_Error_loss, load_cropping_ckpt
from e2eflow.core.keras_util import MetricsHistory, step_decay, plotHistory, plotHistory_val
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ===============================================================================
# Example
# ===============================================================================


# ===============================================================================
# Model Parameters
# ===============================================================================
# read slice info

info_file = "/mnt/data/rawdata/MoCo/LAPNet/resp/slice_info_resp.ods"
ods = get_data(info_file)
slice_info = {value[0]: list(range(*[int(j) - 1 for j in value[1].split(',')])) for value in ods["Sheet1"] if
              len(value) is not 0}
# flow_augment_type: 'constant', 'smooth', 'real_simulated', 'real_simulated_x_smooth'
flow_augment_type = [0, 0.4, 0.2, 0.4]
# flow_augment_type = [0.25, 0.25, 0.25, 0.25]
# flow_augment_type = [1, 0, 0, 0]
# Model
# parameters
crop_size = 33
batch_size = 64
# downsampling type
mask_type = 'drUS'
# mask_type = 'radial'
# mask_type = 'crUS'

params = {'batch_size': batch_size,
          'crop_size': crop_size,
          'mask_type': mask_type,
          'amp': 10,
          'us_rate': 'random',
          'augment_type_percent': flow_augment_type,
          'slice_info': slice_info}

max_epochs = 2

# ===============================================================================
# Dataset path
# ===============================================================================
# Load partitions
# normalized data

TrainingPath = '/mnt/data/rawdata/MoCo/LAPNet/resp/new_data/npz/train/'
list_IDs_train = [TrainingPath + f for f in listdir(TrainingPath) if isfile(join(TrainingPath, f))]
TestPath = '/mnt/data/rawdata/MoCo/LAPNet/resp/new_data/npz/test/'
list_IDs_test = [TestPath + f for f in listdir(TestPath) if isfile(join(TestPath, f))]

# non-normalized data
"""
ImgPath = '/mnt/data/rawdata/MoCo/LAPNet/preprocessed/resp/'
FlowPath = '/mnt/data/rawdata/MoCo/LAPNet/resp/data_with_flow'
list_IDs_train = [os.path.splitext(f)[0] for f in listdir(FlowPath) if isfile(join(FlowPath, f))]
"""
partition = {"train": list_IDs_train[1:],
             "test": list_IDs_test}

# partition = {"train": list_IDs_train[1:],
#             "test": list_IDs_test}

# ===============================================================================
# Data Generator
# ===============================================================================

# Training generator
# training_generator = DataGenerator_Resp_2D(partition['train'], total_data_num=2e+6,**params)
training_generator = DataGenerator_Resp_cropping_2D(partition['train'],
                                                    total_data_num=20,
                                                    shuffle=True,
                                                    **params)
# test samples
X_val, y_val = training_generator.get_data_samples(4)
print(X_val.shape)
print(y_val.shape)
# ===============================================================================
# Model Build + Compile
# ===============================================================================
# test building
build = False
if build:
    # build
    input_shape = (crop_size, crop_size, 4)
    inputs = Input(shape=input_shape, )
    ModelResp = buildLAPNet_model(inputs)
    # show summary
    print(ModelResp.summary())
    # compile
    ModelResp.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, lr=0.0),
                      loss=[End_Point_Error_loss],
                      metrics=['accuracy'])

# ===============================================================================
# Load and evaluate old trained model
# ===============================================================================
# test generator
# test_generator = DataGenerator_Resp_2D(partition['test'], total_data_num=100,**params)
test_generator = DataGenerator_Resp_tapering_2D(partition['test'],
                                                total_data_num=20,
                                                **params)
checkpoint_path_cropping = '/mnt/data/projects/MoCo/LAPNet/UnFlow/log/ex/resp/srx424_drUS_1603/model.ckpt'
# Re-evaluate the model
#loss, acc = ModelResp.evaluate(x=test_generator, max_queue_size=20, workers=6)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# ===============================================================================
# Train or Test the Model
# ===============================================================================
# run
mode_run = 'traiin'
# mode_run = 'test'
if mode_run == 'train':
    # Track accuracy and loss in real-time
    History = MetricsHistory()

    # Checkpoints
    checkpoints = ModelCheckpoint('checkpoints/' + os.path.basename(__file__) + '_{epoch:02d}' + '.hd5f',
                                  save_weights_only=True,
                                  save_freq=5)

    # TensorBoard
    # tb_callback = tf.keras.callbacks.TensorBoard('./logs/run_a', update_freq=1)

    # CSVLogger logs epoch, acc, loss, val_acc, val_loss
    # log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

    # learning rate monitoring
    scheduler = LearningRateScheduler(step_decay)

    # define callbacks
    # callbacks_list = [checkpoints, log_csv, scheduler, History]
    callbacks_list = [scheduler]
    # train
    history = ModelResp.fit_generator(generator=training_generator,
                                      callbacks=callbacks_list,
                                      epochs=max_epochs,
                                      workers=6,
                                      max_queue_size=20,
                                      use_multiprocessing=True)
    # plot training history
    plotHistory(history, max_epochs)
    #plotHistory_val(history, max_epochs)

if mode_run == 'test':
    # Load weights
    weights_path = 'checkpoints/' + os.path.basename(__file__) + '_{0:02d}'.format(max_epochs) + '.hd5f'
    ModelResp.load_weights(weights_path)

# ===============================================================================
# Predictions
# ===============================================================================


# Clear session
K.clear_session()

print('done')
