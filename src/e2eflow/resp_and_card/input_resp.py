import math
import random
import numpy as np
# from ..core.input import Input, load_mat_file
from ..core.image_warp import np_warp_3D
from e2eflow.core.resp_US.sampling import generate_mask
from e2eflow.core.resp_US.sampling_center import sampleCenter
from ..core.util import _u_generation_3D, fftnshift, rectangulartapering2d, flowCrop, arr2kspace, \
    normalize_complex_arr, plottaperedkspace, ifftnshift
import tensorflow.keras as keras
from e2eflow.core.resp_US.retrospective_radial import subsample_radial
from skimage.util.shape import view_as_windows
import os
import scipy.io as sio
import medutils
import matplotlib.pyplot as plt
"""
class MRI_Resp_2D(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def load_aug_data(self,
                      fn_im_paths,
                      slice_info,
                      aug_type,
                      amp=5,
                      mask_type='drUS',
                      US_rate='random',
                      num_to_take=1500):
        
        #:param fn_im_paths: list, the subject list for training
        #:param slice_info: list, which slices to take
        #:param aug_type: synthetic motion augmentation type
        #:param amp: amplitude of augmented motion
        #:param mask_type: currently only 2D radial is available
        #:param US_rate:
        #:param num_to_take:
        #:return:
        

        output = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        if num_to_take == 0:
            return output
        i = 0
        flag = 0
        if num_to_take == 'all':
            num_to_take = 10000000
            flag = 1
        while len(output) <= num_to_take:
            fn_im_path = fn_im_paths[i]
            # fn_im_path = '/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/new_data/npz/train/volunteer_01_cw.npz'
            try:
                f = load_mat_file(fn_im_path)
            except:
                try:
                    f = np.load(fn_im_path)
                except ImportError:
                    print("Wrong Data Format")

            name = fn_im_path.split('/')[-1].split('.')[0]

            ref = np.asarray(f['dFixed'], dtype=np.float32)
            ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
            uy = np.asarray(f['uy'], dtype=np.float32)
            uz = np.zeros(np.shape(ux), dtype=np.float32)
            u = np.stack((ux, uy, uz), axis=-1)
            if aug_type == 'real_x_smooth':
                u_syn = _u_generation_3D(np.shape(ux), amp, motion_type=1)
                u = np.multiply(u, u_syn)
            elif aug_type == 'smooth':
                u = _u_generation_3D(np.shape(ux), amp, motion_type=1)
            elif aug_type == 'constant':
                u = _u_generation_3D(np.shape(ux), amp, motion_type=0)
            elif aug_type == 'real':
                pass
            else:
                raise ImportError('wrong augmentation type is given')
            mov = np_warp_3D(ref, u)

            # # for showing of arrows
            # im1 = ref[..., 35]
            # im2 = mov[..., 35]
            # u = u[..., :2][..., 35, :]
            # save_imgs_with_arrow(im1, im2, u)
            
            if US_rate:
                if US_rate == 'random':
                    if mask_type == 'drUS':
                        acc = np.random.choice(np.arange(1, 32, 6))  # TODO the US-rate can be modified here
                    elif mask_type == 'crUS':
                        acc = np.random.choice(np.arange(1, 15, 4))
                    elif mask_type == 'radial':
                        acc = np.random.choice(np.arange(1, 18, 4))
                else:
                    try:
                        acc = US_rate
                    except ImportError:
                        print("Wrong undersampling rate is given")
                        continue

                if mask_type == 'drUS':
                    mask = np.transpose(generate_mask(acc=acc, size_y=256, nRep=4), (2, 1, 0))
                elif mask_type == 'crUS':
                    mask = sampleCenter(1 / acc * 100, 256, 72)
                    mask = np.array([mask, ] * 4, dtype=np.float32)
                k_ref = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
                k_mov = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
                ref = (np.fft.ifftn(k_ref)).real
                mov = (np.fft.ifftn(k_mov)).real
           

            # save_img(mov[..., 40], '/home/jpa19/PycharmProjects/MA/UnFlow/output/temp/'+'us_mov30')
            # save_img(mask[3, ...], '/home/jpa19/PycharmProjects/MA/UnFlow/output/temp/' + 'mask_mov30')

            data_3D = np.stack((ref, mov, u[..., 0], u[..., 1]), axis=-1)
            data_3D = np.moveaxis(data_3D, 2, 0)
            slice2take = slice_info[name]
            Imgs = data_3D[slice2take, ...]

            # fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            # plt.axis('off')
            # ax[0].imshow(Imgs[1,...,0], cmap='gray')
            # ax[1].imshow(Imgs[1,...,1], cmap='gray')
            # plt.show()

            output = np.concatenate((output, Imgs), axis=0)
            i += 1
            if i == len(fn_im_paths):
                if flag == 0:
                    i = 0
                else:
                    break
        print("{} real {} data are generated".format(num_to_take, aug_type))

        return np.asarray(output[:num_to_take, ...], dtype=np.float32)
    
    def input_train_data(self, img_dirs, slice_info, params, case='train'):
        # strategy 1: fixed total number
        if case == 'train':
            total_data_num = params.get('total_data_num')
        elif case == 'validation':
            total_data_num = 128
        num_constant = math.floor(total_data_num * params.get('augment_type_percent')[0])
        num_smooth = math.floor(total_data_num * params.get('augment_type_percent')[1])
        num_real = math.floor(total_data_num * params.get('augment_type_percent')[2])
        num_real_x_smooth = math.floor(total_data_num * params.get('augment_type_percent')[3])
        assert (num_real <= 1585 and case == 'train') or (num_real <= 136 and case == 'validation')

        # # strategy 2: use all real_simulated data
        # num_real = 1721
        # num_constant = int(params.get('augment_type_percent')[0]/params.get('augment_type_percent')[2]*num_real)
        # num_smooth = int(params.get('augment_type_percent')[1] / params.get('augment_type_percent')[2] * num_real)
        # num_real_x_smooth = int(params.get('augment_type_percent')[3] / params.get('augment_type_percent')[2] * num_real)


        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        fn_im_paths = self.get_data_paths(img_dirs)
        aug_data_constant = self.load_aug_data(fn_im_paths,
                                               slice_info,
                                               aug_type='constant',
                                               amp=params.get('flow_amplitude'),
                                               mask_type=params.get('mask_type'),
                                               US_rate=(params.get('us_rate')),
                                               num_to_take=num_constant)

        np.random.shuffle(fn_im_paths)
        aug_data_smooth = self.load_aug_data(fn_im_paths,
                                             slice_info,
                                             aug_type='smooth',
                                             amp=params.get('flow_amplitude'),
                                             mask_type=params.get('mask_type'),
                                             US_rate=(params.get('us_rate')),
                                             num_to_take=num_smooth)
        # np.save('/home/jpa19/PycharmProjects/MA/UnFlow/hardcoded_data_USfalse', aug_data_smooth)
        np.random.shuffle(fn_im_paths)
        aug_data_real = self.load_aug_data(fn_im_paths,
                                           slice_info,
                                           aug_type='real',
                                           amp=params.get('flow_amplitude'),
                                           mask_type=params.get('mask_type'),
                                           US_rate=(params.get('us_rate')),
                                           num_to_take=num_real)

        np.random.shuffle(fn_im_paths)
        aug_data_real_x_smooth = self.load_aug_data(fn_im_paths,
                                                    slice_info,
                                                    aug_type='real_x_smooth',
                                                    amp=5,
                                                    mask_type=params.get('mask_type'),
                                                    US_rate=(params.get('us_rate')),
                                                    num_to_take=num_real_x_smooth)

        batches = np.concatenate((batches, aug_data_real, aug_data_constant, aug_data_smooth, aug_data_real_x_smooth), axis=0)
        np.random.shuffle(batches)
        if params.get('network') == 'lapnet':
            radius = int((params.get('crop_size') - 1) / 2)
            if params.get('padding'):
                batches = np.pad(batches, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
            if params.get('random_crop'):
                batches = self.crop2D(batches, crop_size=params.get('crop_size'), box_num=1, cut_margin=20)
                # batches = self.taper2D(batches, crop_size=33, box_num=1, cut_margin=20)
            else:
                x_dim, y_dim = np.shape(batches)[1:3]
                pos = pos_generation_2D(intervall=[[0, x_dim - params.get('crop_size') + 1],
                                                   [0, y_dim - params.get('crop_size') + 1]], stride=4)
                batches = self.taper2D_FixPts(batches, crop_size=params.get('crop_size'), box_num=2, pos=pos)
                #batches = self.crop2D_FixPts(batches, crop_size=params.get('crop_size'), box_num=2, pos=pos) #box_num=np.shape(pos)[1]
            #batches = np.concatenate((arr2kspace(batches[..., :2]), batches[..., 2:]), axis=-1)
            im1 = batches[..., :2]
            im2 = batches[..., 2:4]
            flow = batches[:, radius, radius, 4:6]
        elif params.get('network') == 'flownet':
            im1 = batches[..., 0]
            im2 = batches[..., 1]
            flow = batches[..., 2:]
        else:
            raise ImportError('Wrong Network name is given')
        return [im1, im2, flow]

"""


class DataGenerator_Resp_2D(keras.utils.Sequence):
    """
     Generates batches containing the reference k-space and the moved k-space seperated in real and imaginary parts
     as well the corresponding flow
     To be passed as argument in the fit_generator function of Keras.

     Parameters
     ----------
     batch_size : int (optional, default is 32)
        Number of samples in output array of each iteration of the 'generate'
        method.
     img_directory : string (optional, default points to 'files/precomputed')
        Path of the precomputed files.
     crop_size : size of the cropping
     shuffle : boolean (optional, default is True)
        If True, shuffles order of exploration.
    """

    def __init__(self, list_IDs, batch_size,  crop_size, augment_type_percent, slice_info, us_rate, amp, mask_type,
                 total_data_num=1, pos=None, cut_margin=0, shuffle=False, normalized=True):
        """Initialization"""
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.mask_type = mask_type
        self.shuffle = shuffle
        self.cut_margin = cut_margin
        self.crop_size = crop_size
        self.augment_type_percent = augment_type_percent
        self.slice_info = slice_info
        self.us_rate = us_rate
        self.amp = amp
        self.total_data_num = total_data_num
        self.pos = pos
        self.normalized = normalized
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.total_data_num) / self.batch_size)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = random.choices(range(len(self.list_IDs)), k=self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        rest_data_to_be_generated = self.total_data_num - index * self.batch_size
        if rest_data_to_be_generated < self.batch_size:
            self.indexes = self.indexes[:rest_data_to_be_generated]
        # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        # Generate data
        k_tapered, flow = self.data_augmentation(list_IDs_temp)
        return k_tapered, flow

    def data_augmentation(self, list_IDs_temp):
        """Returns augmented data with batch_size kspace"""  # k_scaled : (n_samples, crop_size, crop_size, 4)
        # Initialization
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 4), dtype=np.float32)
        flow = np.empty((self.batch_size, 2), dtype=np.float32)
        return k_out, flow

    def get_data_samples(self, validation_num):
        """Generate a number of validation data"""
        self.indexes_val = random.choices(range(len(self.list_IDs)), k=validation_num)
        list_IDs_temp_val = [self.list_IDs[k] for k in self.indexes_val]
        val_k, val_flow = self.data_augmentation(list_IDs_temp_val)
        return val_k, val_flow

    def downsample_data(self, ref, mov):
        if self.us_rate:
            if self.us_rate == 'random':
                if self.mask_type == 'drUS':
                    acc = np.random.choice(np.arange(1, 32, 6))  # TODO the US-rate can be modified here
                elif self.mask_type == 'crUS':
                    acc = np.random.choice(np.arange(1, 15, 4))
                elif self.mask_type == 'radial':
                    acc = np.random.choice(np.arange(1, 18, 4))
            else:
                try:
                    acc = self.us_rate
                except ImportError:
                    print("Wrong undersampling rate is given")
                    pass
            if self.mask_type == 'radial':
                ref_mov = np.stack((ref, mov), axis=-1)
                ref_mov = np.ascontiguousarray(ref_mov)
                ref_mov_downsampled = subsample_radial(ref_mov, acc, None)
                ref = ref_mov_downsampled[..., 0]
                mov = ref_mov_downsampled[..., 1]
            else:
                if self.mask_type == 'drUS':
                    mask = np.transpose(generate_mask(acc=acc, size_y=256, nRep=4), (2, 1, 0))
                elif self.mask_type == 'crUS':
                    mask = sampleCenter(1 / acc * 100, 256, 72)
                    mask = np.array([mask, ] * 4, dtype=np.float32)
                k_ref = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
                k_mov = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
                ref = (np.fft.ifftn(k_ref)).real
                mov = (np.fft.ifftn(k_mov)).real
        return ref, mov

    def flow_variation(self, ux, u_full, aug_type):
        if aug_type == 'real_x_smooth':
            u_syn = _u_generation_3D(np.shape(ux), self.amp, motion_type=1)
            u = np.multiply(u_full, u_syn)
        elif aug_type == 'smooth':
            u = _u_generation_3D(np.shape(ux), self.amp, motion_type=1)
        elif aug_type == 'constant':
            u = _u_generation_3D(np.shape(ux), self.amp, motion_type=0)
        elif aug_type == 'real':
            u = u_full
        else:
            raise ImportError('wrong augmentation type is given')
        return u

    def Data_3D_to_2D(self, ref, mov, u, layer):
        data_3D = np.stack((ref, mov, u[..., 0], u[..., 1]), axis=-1)
        data_3D = np.moveaxis(data_3D, 2, 0)
        Imgs = data_3D[layer, ...]
        ref_out = Imgs[..., 0]
        mov_out = Imgs[..., 1]
        u_out = Imgs[..., 2:]
        return ref_out, mov_out, u_out

    def find_layer(self, dataID):
        ID = os.path.splitext(os.path.basename(dataID))[0]
        slice = np.random.choice(self.slice_info[ID])
        return slice

    def load_data_3D(self, dataID, aug_type):
        # read data
        ref_3D = self.load_scaled_ref_img(dataID)
        ux, u_full = self.load_scaled_u(dataID)
        # augment data
        u_3D = self.flow_variation(ux, u_full, aug_type)
        # warping
        if self.normalized:
            mov_3D = np_warp_3D(ref_3D, u_3D)
        else:
            mov_3D_real = np_warp_3D(ref_3D.real, u_3D)
            mov_3D_img = np_warp_3D(ref_3D.imag, u_3D)
            mov_3D = mov_3D_real + 1j * mov_3D_img
        # downsample data
        ref_3D, mov_3D = self.downsample_data(ref_3D, mov_3D)
        return ref_3D, mov_3D, u_3D

    def load_data_2D(self, dataID, aug_type):
        """reads data in DataID, create flow as given in aug_type, downsample 3D data, choose a slice"""
        ref_3D, mov_3D, u_3D = self.load_data_3D(dataID, aug_type)
        # chose slice
        slice = self.find_layer(dataID)
        ref_out, mov_out, u_out = self.Data_3D_to_2D(ref_3D, mov_3D, u_3D, slice)
        return ref_out, mov_out, u_out

    def get_aug_type(self, idx):
        typ = 'constant'
        idx_constant = math.floor(self.batch_size * self.augment_type_percent[0]) - 1
        idx_smooth = math.floor(self.batch_size * self.augment_type_percent[1]) + idx_constant
        idx_real = math.floor(self.batch_size * self.augment_type_percent[2]) + idx_smooth

        if idx_constant < idx and (idx <= idx_smooth):
            typ = 'smooth'
        elif (idx > idx_smooth) and (idx <= idx_real):
            typ = 'real'
        elif idx > idx_real:
            typ = 'real_x_smooth'
        return typ

    def load_scaled_ref_img(self, path):
        if self.normalized:
            temp_data = np.load(path)
            res = np.asarray(temp_data['dFixed'], dtype=np.float32)
        else:
            ImgPath = '/mnt/data/rawdata/MoCo/LAPNet/preprocessed/resp/'
            temp_data = sio.loadmat(ImgPath + path + '_img.mat')
            arr = temp_data['dImgC'][:, :, :, 0]
            res = normalize_complex_arr(arr)
        return res

    def load_scaled_u(self, path):
        if self.normalized:
            temp_data = np.load(path)
        else:
            FlowPath = '/mnt/data/rawdata/MoCo/LAPNet/resp/data_with_flow/'
            temp_data = sio.loadmat(FlowPath + path + '.mat')
        ux = np.asarray(temp_data['ux'], dtype=np.float32)
        uy = np.asarray(temp_data['uy'], dtype=np.float32)
        uz = np.zeros(np.shape(ux), dtype=np.float32)
        u = np.stack((ux, uy, uz), axis=-1)
        return ux, u


class DataGenerator_Resp_tapering_2D(DataGenerator_Resp_2D):

    def __init__(self, list_IDs, batch_size, crop_size, augment_type_percent, slice_info, us_rate, amp, mask_type,
                 total_data_num=1, pos=None, cut_margin=0, normalized=True, shuffle=False):
        """Initialization"""
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.mask_type = mask_type
        self.shuffle = shuffle
        self.cut_margin = cut_margin
        self.crop_size = crop_size
        self.augment_type_percent = augment_type_percent
        self.slice_info = slice_info
        self.us_rate = us_rate
        self.amp = amp
        self.total_data_num = total_data_num
        self.pos = pos
        self.normalized = normalized
        self.on_epoch_end()

    def taper2D(self, ref_img, mov_img, u):
        """taper"""
        x_dim, y_dim = np.shape(ref_img)[0:2]
        k_ref_full = fftnshift(ref_img)
        k_mov_full = fftnshift(mov_img)
        if self.pos is None:
            x_pos = np.random.randint(0 + self.cut_margin, x_dim - self.crop_size + 1 - self.cut_margin)
            y_pos = np.random.randint(0 + self.cut_margin, y_dim - self.crop_size + 1 - self.cut_margin)
        else:
            x_pos = self.pos[0]
            y_pos = self.pos[1]
        k_ref = rectangulartapering2d(k_ref_full, x_pos, y_pos, self.crop_size)
        k_mov = rectangulartapering2d(k_mov_full, x_pos, y_pos, self.crop_size)
        flow = flowCrop(u, x_pos, y_pos, self.crop_size)
        return k_ref, k_mov, flow

    def data_augmentation(self, list_IDs_temp):
        """Returns augmented data with batch_size kspace"""  # k_scaled : (n_samples, crop_size, crop_size, 4)
        # Initialization
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 4), dtype=np.float32)
        flow = np.empty((self.batch_size, 2), dtype=np.float32)

        # Computations

        for i, ID in enumerate(list_IDs_temp):
            aug = self.get_aug_type(i)
            ref_img, mov_img, u = self.load_data_2D(ID, aug)
            # tapering
            k_out[i, :, :, :2], k_out[i, :, :, 2:4], flow[i, ...] = self.taper2D(ref_img, mov_img, u)
        return k_out, flow


class DataGenerator_Resp_cropping_2D(DataGenerator_Resp_2D):

    def __init__(self, list_IDs, batch_size, crop_size, augment_type_percent, slice_info, us_rate, amp, mask_type,
                 total_data_num=1, pos=None, cut_margin=0, shuffle=False, normalized=True):
        """Initialization"""
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.mask_type = mask_type
        self.shuffle = shuffle
        self.cut_margin = cut_margin
        self.crop_size = crop_size
        self.augment_type_percent = augment_type_percent
        self.slice_info = slice_info
        self.us_rate = us_rate
        self.amp = amp
        self.total_data_num = total_data_num
        self.pos = pos
        self.normalized = normalized
        self.on_epoch_end()

    def crop2D(self, ref, mov, u):
        """crop a given array in time domain"""
        # padding
        # radius = int((self.crop_size - 1) / 2)
        # ref_tmp = np.pad(ref, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
        # mov_tmp = np.pad(mov, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
        # u_tmp = np.pad(u, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

        # random position
        x_dim, y_dim = np.shape(ref)
        if self.pos is None:
            x_pos = np.random.randint(0 + self.cut_margin, x_dim - self.crop_size + 1 - self.cut_margin)
            y_pos = np.random.randint(0 + self.cut_margin, y_dim - self.crop_size + 1 - self.cut_margin)
        else:
            x_pos = self.pos[0]
            y_pos = self.pos[1]

        # cropping
        window_size=(self.crop_size, self.crop_size)
        ref_tmp = view_as_windows(ref, window_size)[x_pos,y_pos]
        mov_tmp = view_as_windows(mov, window_size)[x_pos, y_pos]
        ref_mov = np.stack((ref_tmp, mov_tmp), axis=-1)
        flow_out = flowCrop(u, x_pos, y_pos, self.crop_size)

        return ref_mov, flow_out

    def data_augmentation(self, list_IDs_temp):
        # Initialization
        k_out = np.empty((self.batch_size, self.crop_size, self.crop_size, 2), dtype=np.float32)
        flow = np.empty((self.batch_size, 2), dtype=np.float32)
        # Computations
        for i, ID in enumerate(list_IDs_temp):
            aug = self.get_aug_type(i)
            ref_tmp, mov_tmp, u_tmp = self.load_data_2D(ID, aug)
            k_out[i, ...], flow[i, ...] = self.crop2D(ref_tmp, mov_tmp, u_tmp)
        k_end = arr2kspace(k_out)
        return k_end, flow
