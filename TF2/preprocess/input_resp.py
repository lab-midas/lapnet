import numpy as np
from core.cropping import arr2kspace, crop2D
from core.tapering import taper2D
import tensorflow.keras as keras
import os
from random import shuffle


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
        """Returns augmented data with batch_size kspace"""  # k_scaled : (n_samples, crop_size, crop_size, 4)
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

        return [X[..., :2], X[..., 2:]], X[..., :2]

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Generate data
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 4), dtype=np.float32)
        flow = np.empty((self.batch_size, 2), dtype=np.float32)
        # Generate data
        for i in range(len(list_IDs_temp)):
            try:
                ind = list_IDs_temp[i][1]
                data = np.load(list_IDs_temp[i][0], mmap_mode='r')
                # data = np.load(list_IDs_temp[i], mmap_mode='r')
                k_out[i, ...] = data['k_space'][ind, ...]
                #k_out[i, ...] = data['train_kspace'][..., 0]
                flow[i, ...] = data['flow'][ind, ...]
                # flow[i, ...] = data['flow'][:2]
            except:
                pass

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
            try:
                data = np.load(list_IDs_temp[i])
                k_all[i, ...] = data['train_kspace']
                flow[i, ...] = data['flow']
            except:
                pass

        """ux = flow[:, :2]
        uy = np.stack((flow[:, 0], flow[:, 2]), axis=-1)
        uz = flow[:, 1:]"""

        k_cor = k_all[..., 0]
        k_sag = k_all[..., 1]
        # k_ax = k_all[..., 2]

        return k_cor, k_sag, flow
