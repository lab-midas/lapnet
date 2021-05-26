import math
import tensorflow as tf
from TF2.preprocess.input_resp import DataGenerator_3D, DataGenerator_2D, DataGenerator_Resp_train_2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os


def step_decay(epoch):
    initial_lr = 2.5e-04
    drop = 0.5
    epochs_drop = 2
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


# EPE function modified
def modified_EPE(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    final_loss = tf.reduce_mean(squared_difference)
    return final_loss


def EPE(flows_gt, flows):
    # Given ground truth and estimated flow must be unscaled
    return tf.reduce_mean(tf.norm(flows_gt - flows, ord=2, axis=0))


def EAE(y_true, y_pred):
    final_loss = tf.math.real(tf.math.acos((1 + tf.reduce_sum(tf.math.multiply(y_true, y_pred))) /
                                           (tf.math.sqrt(1 + tf.reduce_sum(tf.math.pow(y_pred, 2))) *
                                            tf.math.sqrt(1 + tf.reduce_sum(tf.math.pow(y_true, 2))))))
    return final_loss


def LAP_loss_function(y_true, y_pred):
    w_1 = 0.8
    w_2 = 0.2
    return tf.add(w_1 * modified_EPE(y_true, y_pred), w_2 * EAE(y_true, y_pred))


def train(ModelResp, general_setup, experiment_setup):
    # setup
    slicing_mode = general_setup['slicing_mode']
    dimensionality = general_setup['dimensionality']
    logs_path = experiment_setup['logs_path']

    # compile
    ModelResp.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, lr=0.0),
                      loss=LAP_loss_function,
                      metrics=['accuracy'])

    # Model Parameters
    experiment_name = experiment_setup['experiment_name']
    checkpoint_file = experiment_setup['checkpoint_file']
    if not os.path.exists(f'{logs_path}/checkpoints/{checkpoint_file}'):
        os.makedirs(f'{logs_path}/checkpoints/{checkpoint_file}')
    if not os.path.exists(f'{logs_path}/graphs'):
        os.makedirs(f'{logs_path}/graphs')

    max_epochs = experiment_setup['num_epochs']
    batch_size = experiment_setup['batch_size']
    num_workers = experiment_setup['num_workers']

    # generator
    TrainingPath = experiment_setup['data_path']
    if slicing_mode == 'tapering':
        if dimensionality == '2D':
            training_generator = DataGenerator_2D(TrainingPath, batch_size=batch_size)
        if dimensionality == '3D':
            training_generator = DataGenerator_3D(TrainingPath, batch_size=batch_size)

    if slicing_mode == 'cropping':
        training_generator = DataGenerator_Resp_train_2D(TrainingPath, batch_size=batch_size)

    if experiment_setup['weights_path']:
        weights_path = experiment_setup['weights_path']
        ModelResp.load_weights(weights_path)

    # Checkpoints
    checkpoints = ModelCheckpoint(
        f'{logs_path}/checkpoints/{checkpoint_file}/{experiment_name}_' + '{epoch:02d}' + '.hd5f',
        save_weights_only=True,
        save_freq="epoch")

    # TensorBoard
    """tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{logs_path}/graphs/', update_freq='5000')"""

    # learning rate monitoring
    scheduler = LearningRateScheduler(step_decay)

    # define callbacks
    callbacks_list = [scheduler, checkpoints]

    # train
    ModelResp.fit_generator(generator=training_generator,
                            callbacks=callbacks_list,
                            verbose=1,
                            epochs=max_epochs,
                            workers=num_workers,
                            max_queue_size=20,
                            use_multiprocessing=True)
