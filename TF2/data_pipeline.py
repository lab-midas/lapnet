import numpy as np
from core.cropping import arr2kspace, crop2D
from core.tapering import taper2D, ifft_along_dim
import tensorflow.keras as keras
import os
from random import shuffle
import tensorflow as tf


class DataRead():
    def __init__(self, height=256, width=256, crop_size=33):
        self.H = height
        self.W = width
        self.crop_size = crop_size
        self.image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'flow': tf.io.FixedLenFeature([], tf.string),
            'image_ref_real': tf.io.FixedLenFeature([], tf.string),
            'image_ref_imag': tf.io.FixedLenFeature([], tf.string),
            'image_mov_real': tf.io.FixedLenFeature([], tf.string),
            'image_mov_imag': tf.io.FixedLenFeature([], tf.string),
        }


    def fetch_2D_data(self, element):
        content =  tf.io.parse_single_example(element, self.image_feature_description)
        image_ref_real = content['image_ref_real']
        image_ref_imag = content['image_ref_imag']
        image_mov_real = content['image_mov_real']
        image_mov_imag = content['image_mov_imag']
        full_flow = content['flow']

        image_ref_real = tf.io.parse_tensor(image_ref_real, out_type=tf.float32)
        image_ref_imag = tf.io.parse_tensor(image_ref_imag, out_type=tf.float32)
        image_mov_real = tf.io.parse_tensor(image_mov_real, out_type=tf.float32)
        image_mov_imag = tf.io.parse_tensor(image_mov_imag, out_type=tf.float32)
        full_flow = tf.io.parse_tensor(full_flow, out_type=tf.float32)

        shape = tf.shape((self.H, self.W))
        image_ref_real = tf.reshape(image_ref_real, shape)
        image_ref_imag = tf.reshape(image_ref_imag, shape)
        image_mov_real = tf.reshape(image_mov_real, shape)
        image_mov_imag = tf.reshape(image_mov_imag, shape)
        full_flow = tf.reshape(full_flow, (self.H, self.W, 2))

        image_ref = image_ref_real +1j * image_ref_imag
        image_mov = image_mov_real + 1j * image_mov_imag

        x_pos = np.random.randint(0, self.H - self.crop_size + 1)
        y_pos = np.random.randint(0, self.W - self.crop_size + 1)

        k_ref, k_mov, flow = taper2D(image_ref, image_mov, x_pos, y_pos, crop_size=self.crop_size, u=full_flow)
        k_space = tf.stack((k_ref, k_mov), axis=-1)

        return k_space, flow


class DataGenerator_Resp_train_2D(keras.utils.Sequence):
    """
     Generates batches containing the reference k-space and the moved k-space seperated in real and imaginary parts
     as well the corresponding flow
     to be passed as argument in the fit function of Keras.

     Two modes are possible:
        - cropping
        - tapering


     Example
     -------
     from e2eflow.preprocess.input_resp import DataGenerator_Resp_train_2D

     # initialize data generator for raw data
     data_path = '/mnt/data/rawdata/MoCo/LAPNet/resp/LAP'
     cropping_training_generator = DataGenerator_Resp_train_2D(data_path)
    """

    def __init__(self, path, crop_size=33, batch_size=64, box_num=200, pos=None, mode='cropping', cut_margin=0,
                 shuffle=False):
        """Initialization"""
        self.batch_size = batch_size
        self.path = path
        self.shuffle = shuffle
        self.cut_margin = cut_margin
        self.crop_size = crop_size
        self.box_num = box_num
        self.pos = pos
        self.mode = mode
        self.list_IDs = self.augment_path()
        self.pos_list = self.get_pos()
        self.on_epoch_end()

    def augment_path(self):
        res = [path for path in self.list_IDs for i in range(self.box_num)]
        shuffle(res)
        return res

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs_aug[k] for k in indexes]
        positions = self.pos_list[indexes, :]
        # Generate data
        k_tapered, flow = self.data_augmentation(list_IDs_temp, positions)
        return k_tapered, flow

    def load_data_2D(self, ID):
        data = np.load(ID)
        ref = data['ref']
        mov = data['mov']
        u = data['flow']
        return ref, mov, u

    def get_pos(self, x_dim=256, y_dim=256):
        x_pos = np.random.randint(0 + self.cut_margin, x_dim - self.crop_size + 1 - self.cut_margin,
                                  len(self.list_IDs_aug))
        y_pos = np.random.randint(0 + self.cut_margin, y_dim - self.crop_size + 1 - self.cut_margin,
                                  len(self.list_IDs_aug))
        res = np.stack((x_pos, y_pos), axis=-1)
        return res

    def data_augmentation(self, list_IDs_temp, positions):
        """Returns augmented data with batch_size kspace"""
        # Initialization
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 4), dtype=np.float32)
        img = np.empty((self.batch_size, self.crop_size, self.crop_size, 2), dtype=np.float32)
        flow = np.empty((self.batch_size, 2), dtype=np.float32)

        # Computations
        for i, ID in enumerate(list_IDs_temp):
            ref_img, mov_img, u = self.load_data_2D(ID)
            pos = positions[i, :]
            if self.mode == 'tapering':
                # tapering
                k_out[i, :, :, :2], k_out[i, :, :, 2:4], flow[i, ...] = taper2D(ref_img,
                                                                                mov_img,
                                                                                u,
                                                                                pos,
                                                                                self.crop_size)
            elif self.mode == 'cropping':
                # cropping
                img[i, ...], flow[i, ...] = crop2D(ref_img,
                                                   mov_img,
                                                   u,
                                                   pos,
                                                   self.crop_size)
        if self.mode == 'cropping':
            k_out = arr2kspace(img)

        return k_out, flow


class DataGenerator_2D(keras.utils.Sequence):
    """
         Generates batches containing the reference k-space and the moved k-space seperated in real and imaginary parts
         from already preprocessed tapered data as well the corresponding flow to be passed as argument in the fit
         function of Keras.


    Example
    -------
     from e2eflow.preprocess.input_resp import DataGenerator_2D

     # initialize data generator for 2D preprocessed data
    preprocessed_data_path = '/scratch/LAPNet/2D_dataset'
    tapering_training_generator_2D = DataGenerator_2D(preprocessed_data_path)
    """

    def __init__(self, Dataset_path, loss_type=None, batch_size=64, num_data=1.5e+6, aug_type=None, shuffle=True, crop_size=33,
                 box_num=200, training_mode='supervised'):
        """Initialization"""
        if aug_type is None:
            aug_type = [0.2, 0.4, 0.4]
        self.aug_type = aug_type
        self.dataset_path: str = Dataset_path
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.num_data: int = num_data
        self.box_num: int = box_num
        self.crop_size: int = crop_size
        self.loss_type = loss_type
        self.list = self.get_paths_list()
        self.training_mode = training_mode
        self.on_epoch_end()

    def make_list_samples(self):
        list_IDs = [x for x in os.listdir(self.dataset_path)]
        res = [(os.path.join(self.dataset_path, path), i) for path in list_IDs for i in range(self.box_num)]
        shuffle(res)
        return res


    def get_paths_list(self):
        num_real = int(self.num_data * self.aug_type[0])
        num_realxsmooth = int(self.num_data * self.aug_type[1])
        num_smooth = int(self.num_data * self.aug_type[2])
        num_list = [num_real, num_smooth, num_realxsmooth]
        res = []
        list_dir = [x[0] for x in os.walk(self.dataset_path)][1:4]
        for i, mypath in enumerate(list_dir):
            list_tmp = [(f'{mypath}/{item}', box) for item in os.listdir(mypath) for box in range(self.box_num)]
            shuffle(list_tmp)
            res += list_tmp[:num_list[i]]
            shuffle(res)

        return res

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Generate data
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 4), dtype=np.float32)
        if self.training_mode is 'self_supervised':
            img_ref = np.empty((self.batch_size,2 , self.crop_size, self.crop_size), dtype=np.float32)
            img_mov = np.empty((self.batch_size, 2, self.crop_size, self.crop_size), dtype=np.float32)
            for i in range(len(list_IDs_temp)):
                ind = list_IDs_temp[i][1]
                data = np.load(list_IDs_temp[i][0], mmap_mode='r')
                k_out[i, ...] = data['k_space'][ind, ...]
                cimg_ref = ifft_along_dim(data['k_space'][ind, :, :, 0] + 1j * data['k_space'][ind, :, :, 1])
                img_ref[i, ...] = np.stack((cimg_ref.real, cimg_ref.imag), axis=0)
                cimg_mov = ifft_along_dim(data['k_space'][ind, :, :, 2] + 1j * data['k_space'][ind, :, :, 3])
                img_mov[i, ...] = np.stack((cimg_mov.real, cimg_mov.imag), axis=0)
            return (k_out, img_mov), img_ref

        if self.training_mode is 'supervised':
            flow = np.empty((self.batch_size, 2), dtype=np.float32)
            for i in range(len(list_IDs_temp)):
                ind = list_IDs_temp[i][1]
                data = np.load(list_IDs_temp[i][0], mmap_mode='r')
                k_out[i, ...] = data['k_space'][ind, ...]
                flow[i, ...] = data['flow'][ind, ...]
            return k_out, flow






class DataGenerator_3D(keras.utils.Sequence):
    """
       Generates batches containing the reference k-space and the moved k-space seperated in real and imaginary parts
       for the three diretions from already preprocessed tapered data as well the corresponding 3D flow to be passed as
       argument in the fit function of Keras.

    Example
    -------
    from e2eflow.preprocess.input_resp import DataGenerator_3D

     # initialize data generator for 3D preprocessed data
    preprocessed_3D_data_path = '/scratch/LAPNet/3D_dataset'
    tapering_training_generator_3D = DataGenerator_3D(preprocessed_3D_data_path)
    """

    def __init__(self, Dataset_path, batch_size=64, num_data=1.5e+6, aug_type=None, shuffle=True, crop_size=33,
                 box_num=200):
        """Initialization"""
        if aug_type is None:
            aug_type = [0.2, 0.4, 0.4]
        self.aug_type = aug_type
        self.dataset_path: str = Dataset_path
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.num_data: int = num_data
        self.box_num: int = box_num
        self.crop_size: int = crop_size
        self.list = self.get_paths_list()
        self.on_epoch_end()

    def get_paths_list(self):

        num_real = int(self.num_data * self.aug_type[0])
        num_realxsmooth = int(self.num_data * self.aug_type[1])
        num_smooth = int(self.num_data * self.aug_type[2])
        num_list = [num_real, num_smooth, num_realxsmooth]
        res = []
        list_dir = [x[0] for x in os.walk(self.dataset_path)][1:4]
        for i, mypath in enumerate(list_dir):
            list_tmp = [f'{mypath}/{item}' for item in os.listdir(mypath) if item.endswith('.npz')]
            shuffle(list_tmp)
            res += list_tmp[:num_list[i]]
            shuffle(res)

        return res

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # ind = self.indexes[index]
        # Find list of IDs
        list_IDs_temp = [self.list[k] for k in indexes]
        # Generate data
        k_cor, k_sag, flow = self.__data_generation(list_IDs_temp)

        return (k_cor, k_sag), flow

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Generate data
        k_all = np.zeros((self.batch_size, self.crop_size, self.crop_size, 4, 3), dtype=np.float32)
        flow = np.zeros((self.batch_size, 3), dtype=np.float32)
        # Generate data
        for i in range(len(list_IDs_temp)):
            data = np.load(list_IDs_temp[i])
            k_all[i, ...] = data['train_kspace']
            flow[i, ...] = data['flow']

        k_cor = k_all[..., 0]
        k_sag = k_all[..., 1]
        # k_ax = k_all[..., 2]

        return k_cor, k_sag, flow
