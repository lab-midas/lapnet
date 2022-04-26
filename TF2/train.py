import math
from TF2.data_pipeline import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import json


def step_decay(epoch):
    initial_lr = 1e-03
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


def loss1(y_true, y_pred):
    w = 0.5
    y1 = LAP_loss_function(y_true[0], y_pred[0])
    y2 = w * LAP_loss_function(y_true[1], y_pred[1])
    squared_difference = tf.stack([y1, y2], axis=-1)
    return squared_difference


def loss2(y_true, y_pred):
    w = 0.5
    y1 = w * LAP_loss_function(y_true[0], y_pred[0])
    y2 = LAP_loss_function(y_true[1], y_pred[1])
    squared_difference = tf.stack([y1, y2], axis=-1)
    return squared_difference


def train_supervised(ModelResp, general_setup, experiment_setup):
    # setup
    slicing_mode = general_setup['slicing_mode']
    dimensionality = general_setup['dimensionality']
    logs_path = experiment_setup['logs_path']

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
    training_data_path = experiment_setup['data_path']
    if slicing_mode == 'tapering':
        if dimensionality == '2D':
            # training_generator = DataGenerator_2D(training_data_path, batch_size=batch_size)
            # prepare training data
            TRAINSET_FNAME_real = 'training2Ddata_LAP.tfrecord'
            fname_real = os.path.join(training_data_path, TRAINSET_FNAME_real)

            TRAINSET_FNAME_smooth = 'training2Ddata_smooth.tfrecord'
            fname_smooth = os.path.join(training_data_path, TRAINSET_FNAME_smooth)

            TRAINSET_FNAME_X = 'training2Ddata_X.tfrecord'
            fname_X = os.path.join(training_data_path, TRAINSET_FNAME_X)

            AUTO = tf.data.experimental.AUTOTUNE
            tfrds = tf.data.Dataset.list_files([fname_real, fname_smooth, fname_X] * 200)

            datar = DataRead()
            train_ds = (
                tfrds
                    .shuffle(10000)
                    .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
                    .shuffle(10000)
                    .map(lambda image: datar.fetch_2D_data(image), num_parallel_calls=AUTO)
                    .cache()
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(AUTO)

            )

        if dimensionality == '3D':
            training_generator = DataGenerator_3D(training_data_path, batch_size=batch_size)

    if slicing_mode == 'cropping':
        training_generator = DataGenerator_Resp_train_2D(training_data_path, batch_size=batch_size)

    if experiment_setup['weights_path']:
        weights_path = experiment_setup['weights_path']
        ModelResp.load_weights(weights_path)

    # Checkpoints
    checkpoints = ModelCheckpoint(
        f'{logs_path}/checkpoints/{checkpoint_file}/{experiment_name}_' + '{epoch:02d}' + '.hd5f',
        save_weights_only=True,
        save_freq=2000)

    # TensorBoard
    """tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{logs_path}/graphs/', update_freq='5000')"""

    # learning rate monitoring
    scheduler = LearningRateScheduler(step_decay)

    # define callbacks
    callbacks_list = [scheduler, checkpoints]

    # train_supervised
    ModelResp.fit(x=train_ds,
                  callbacks=callbacks_list,
                  verbose=1,
                  epochs=max_epochs,
                  workers=num_workers,
                  max_queue_size=20,
                  use_multiprocessing=True)


def train_unsupervised(ModelResp, general_setup, experiment_setup):
    # setup
    slicing_mode = general_setup['slicing_mode']
    dimensionality = general_setup['dimensionality']
    logs_path = experiment_setup['logs_path']
    loss_type = experiment_setup['loss']

    loss = tf.keras.losses.MeanAbsoluteError()
    optimizer =  Adam(beta_1=0.9, beta_2=0.999, lr=0.0)


    ModelResp.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

    # Model Parameters
    experiment_name = experiment_setup['experiment_name']
    checkpoint_file = experiment_setup['checkpoint_file']
    checkpoints_path = f'{logs_path}/checkpoints/{checkpoint_file}'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    if not os.path.exists(f'{logs_path}/graphs'):
        os.makedirs(f'{logs_path}/graphs')

    max_epochs = experiment_setup['num_epochs']
    batch_size = experiment_setup['batch_size']
    num_workers = experiment_setup['num_workers']

    # generator
    TrainingDataPath = experiment_setup['data_path']
    if experiment_setup['data_fetch_method'] is 'generator':
        if slicing_mode == 'tapering':
            if dimensionality == '2D':
                train_ds = DataGenerator_2D(TrainingDataPath, batch_size=batch_size, loss_type= loss_type)
            if dimensionality == '3D':
                train_ds = DataGenerator_3D(TrainingDataPath, batch_size=batch_size)

        if slicing_mode == 'cropping':
            train_ds = DataGenerator_Resp_train_2D(TrainingDataPath, batch_size=batch_size)

    elif experiment_setup['data_fetch_method'] is 'tfrecords':
        # prepare training data
        TRAINSET_FNAME_real = 'training2Ddata_LAP.tfrecord'
        fname_real = os.path.join(TrainingDataPath, TRAINSET_FNAME_real)

        TRAINSET_FNAME_smooth = 'training2Ddata_smooth.tfrecord'
        fname_smooth = os.path.join(TrainingDataPath, TRAINSET_FNAME_smooth)

        TRAINSET_FNAME_X = 'training2Ddata_X.tfrecord'
        fname_X = os.path.join(TrainingDataPath, TRAINSET_FNAME_X)

        AUTO = tf.data.experimental.AUTOTUNE
        tfrds = tf.data.Dataset.list_files([fname_real, fname_smooth, fname_X])

        datar = DataRead()
        train_ds = (
            tfrds
                .shuffle(10000)
                .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
                .shuffle(10000)
                .map(lambda image: datar.fetch_2D_data(image), num_parallel_calls=AUTO)
                .cache()
                .batch(batch_size, drop_remainder=True)
                .prefetch(AUTO)

        )


    if experiment_setup['weights_path']:
        weights_path = experiment_setup['weights_path']
        ModelResp.load_weights(weights_path)

    # Checkpoints
    if not os.path.exists(os.path.join(checkpoints_path, experiment_name)):
        os.makedirs(os.path.join(checkpoints_path, experiment_name))
    # save hyperparameters in file
    create_params_list(experiment_setup, f'{checkpoints_path}/{experiment_name}')
    checkpoints = ModelCheckpoint(
        f'{checkpoints_path}/{experiment_name}/{experiment_name}_' + '{epoch:02d}' + '.hd5f',
        save_weights_only=True,
        save_freq=400)

    # learning rate monitoring
    scheduler = LearningRateScheduler(step_decay)

    # define callbacks
    callbacks_list = [checkpoints, scheduler]

    # train_supervised
    ModelResp.fit(x=train_ds,
                  callbacks=callbacks_list,
                  verbose=1,
                  epochs=max_epochs,
                  workers=num_workers,
                  max_queue_size=20,
                  use_multiprocessing=True)

def create_params_list(experiment_setup, checkpoints_path):
    max_epochs = experiment_setup['num_epochs']
    batch_size = experiment_setup['batch_size']
    lr = experiment_setup['learning_rate']
    loss = experiment_setup['loss']
    optimizer = experiment_setup['optimizer']

    data = {}
    data['hyperparams'] = []
    data['hyperparams'].append({
        "model_version": "baseline",
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": max_epochs,
        "loss_function": loss,
        "optimizer": optimizer
    })
    json_path = os.path.join(checkpoints_path, 'hyperparams.txt')
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)

def train_LAPNET_with_data_API(model, train_ds, logs_path, experiment_name):
    # build the model
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, lr=0.0),
                  loss=LAP_loss_function,
                  metrics=['accuracy'])
    # Checkpoints
    checkpoints = ModelCheckpoint(
        f'{logs_path}/checkpoints/{experiment_name}.hd5f',
        save_weights_only=True,
        save_freq="epoch")

    # learning rate monitoring
    scheduler = LearningRateScheduler(step_decay)

    # define callbacks
    callbacks_list = [scheduler, checkpoints]

    # train_supervised
    model.fit(x=train_ds,
              callbacks=callbacks_list,
              verbose=1,
              epochs=10,
              workers=4,
              steps_per_epoch=23000,
              max_queue_size=20,
              use_multiprocessing=True)





