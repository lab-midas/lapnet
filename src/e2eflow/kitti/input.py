import os
import sys
import math
import time

import numpy as np
import tensorflow as tf
import random
import scipy.io as sio
from skimage.transform import warp
import h5py
from skimage.util.shape import view_as_windows
import pylab
import matplotlib.pyplot as plt

from ..core.input import read_png_image, Input
from ..core.augment import random_crop
from ..core.flow_util import flow_to_color
from ..core.image_warp import image_warp


def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    gt_uint16 = tf.image.decode_png(value, dtype=tf.uint16)
    gt = tf.cast(gt_uint16, tf.float32)
    flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
    mask = gt[:, :, 2:3]
    return flow, mask


def np_warp_2D(img, flow):
    img = img.astype('float32')
    flow = flow.astype('float32')
    height, width = np.shape(img)[0], np.shape(img)[1]
    posx, posy = np.mgrid[:height, :width]
    # flow=np.reshape(flow, [-1, 3])
    vx = flow[:, :, 0]
    vy = flow[:, :, 1]

    coord_x = posx + vx
    coord_y = posy + vy
    coords = np.array([coord_x, coord_y])
    warped = warp(img, coords, order=1)  # order=1 for bi-linear
    return warped


def load_mat_file(fn_im_path):
    try:
        f = sio.loadmat(fn_im_path)
    except Exception:
        try:
            f = h5py.File(fn_im_path, 'r')
        except IOError:
            # print("File {} is defective and cannot be read!".format(fn_im_path))
            raise IOError("File {} is defective and cannot be read!".format(fn_im_path))
    return f


def _u_generation_2D(img_size, amplitude, motion_type=0):
    """

    :param img_size:
    :param amplitude:
    :param motion_type: 0: constant, 1: smooth
    :return:
    """
    M, N = img_size
    if motion_type == 0:
        #u_C = 2 * np.random.rand(2)
        u_C = -1 + 2 * np.random.rand(2)  # interval [-1, 1]
        amplitude = amplitude / np.linalg.norm(u_C, 2)
        u = amplitude * np.ones((M, N, 2))
        u[..., 0] = u_C[0] * u[..., 0]
        u[..., 1] = u_C[1] * u[..., 1]
        pass
    elif motion_type == 1:
        u = np.random.normal(0, 1, (M, N, 2))
        cut_off = 0.01
        w_x_cut = math.floor(cut_off / (1 / M) + (M + 1) / 2)
        w_y_cut = math.floor(cut_off / (1 / N) + (N + 1) / 2)

        LowPass_win = np.zeros((M, N))
        LowPass_win[(M - w_x_cut): w_x_cut, (N - w_y_cut): w_y_cut] = 1

        u[..., 0] = (np.fft.ifft2(np.fft.fft2(u[..., 0]) * np.fft.ifftshift(LowPass_win))).real
        # also equal to u[..., 0] =
        # (np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u[..., 0])) * LowPass_win))).real
        u[..., 1] = (np.fft.ifft2(np.fft.fft2(u[..., 1]) * np.fft.ifftshift(LowPass_win))).real

        u1 = u[..., 0].flatten()
        u2 = u[..., 1].flatten()
        amplitude = amplitude / max(np.linalg.norm(np.vstack([u1, u2]), axis=0))
        u = u * amplitude

    elif motion_type == 'realistic':
        pass

    return u


def imgpair2kspace(arr):
    im1_kspace = to_freq_space(arr[..., 0])
    im2_kspace = to_freq_space(arr[..., 1])
    return np.asarray(np.concatenate((im1_kspace, im2_kspace), axis=-1), dtype=np.float32)


def image2kspace(y, normalize=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    x = to_freq_space(y)  # FFT: (256, 256, 2)
    #x = np.expand_dims(x, axis=0)  # (1, 256, 256, 2)
    if normalize:
        x = x - np.mean(x)

    return x


def to_freq_space_tf(img):
    img_f = tf.signal.fft2d(img)  # FFT
    img_fshift = tf.signal.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def to_freq_space(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f, axes=(-1, -2))  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.stack((img_real, img_imag), axis=-1)  # (im_size1, im_size2, 2)

    return img_real_imag


class MRI_Resp_2D(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def get_data_paths(self, img_dirs):
        fn_im_paths = []
        for img_dir in img_dirs:
            img_dir = os.path.join(self.data.current_dir, img_dir)
            img_files = os.listdir(img_dir)
            for img_file in img_files:
                if '.mat' in img_file:
                    im_path = os.path.join(img_dir, img_file)
                    fn_im_paths.append(im_path)
                else:
                    if not img_file.startswith('.'):
                        try:
                            img_mat = os.listdir(os.path.join(img_dir, img_file))[0]
                        except Exception:
                            print("File {} is empty!".format(img_file))
                            continue
                        fn_im_paths.append(os.path.join(img_dir, img_file, img_mat))
        return fn_im_paths

    def crop2D(self, arr, crop_size, box_num, pos=None):
        """

        :param arr:
        :param crop_size:
        :param box_num: crops per slices
        :param pos: shape (2, n), pos_x and pos_y, n must be same as the box number. if pos=None, select random pos
        :return:
        """
        arr_cropped_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, arr.shape[-1]), dtype=np.float32)
        for i in range(box_num):
            if pos is None:
                x_pos = np.random.randint(0, self.dims[0] - crop_size, arr.shape[0])
                y_pos = np.random.randint(0, self.dims[1] - crop_size, arr.shape[0])
            else:
                x_pos = pos[0]
                y_pos = pos[0]

            w = view_as_windows(arr, (1, crop_size, crop_size, 1))[..., 0, :, :, 0]
            out = w[np.arange(arr.shape[0]), x_pos, y_pos]
            out = out.transpose(0, 2, 3, 1)
            arr_cropped_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], ...] = out
        return arr_cropped_augmented

    def load_real_simulated_data(self, fn_im_paths, selected_slices, max_num_to_take=2000):
        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        for fn_im_path in fn_im_paths:
            f = load_mat_file(fn_im_path)
            im1 = f['I1_real'][..., selected_slices]
            # pylab.imshow(im1[:, :, 35])
            im1 = im1[np.newaxis, ...]
            im2 = f['I1_Real_hat'][..., selected_slices]
            im2 = im2[np.newaxis, ...]
            u0 = f['u_Real_est_1'][..., selected_slices]
            u0 = u0[np.newaxis, ...]
            u1 = f['u_Real_est_2'][..., selected_slices]
            u1 = u1[np.newaxis, ...]

            # im1 = np.squeeze(im1[..., 20])
            # im2 = np.squeeze(im2[..., 20])
            # u0 = u0[..., 20]
            # u1 = u1[..., 20]
            # u = np.concatenate((u0, u1), axis=0)
            # u = np.swapaxes(u, 0, 2)
            # im1_hat = np_warp_2D(im2, -u)
            # im1_hat_hat = np_warp_2D(im2, u)
            # ori_error = im1 - im2
            #
            # warped_error_1 = im1 - im1_hat
            # warped_error_2 = im1 - im1_hat_hat
            # fig, ax = plt.subplots(1, 3, figsize=(15, 8))
            #
            # ax[0].imshow(ori_error, cmap='gray')  # ref
            #
            # ax[1].imshow(warped_error_1, cmap='gray')  # mov
            #
            # ax[2].imshow(warped_error_2, cmap='gray')  # mov
            #
            #
            # pylab.imshow(ori_error, cmap='gray')
            # pylab.figure()
            # pylab.imshow(warped_error, cmap='gray')

            batch = np.concatenate((im1, im2, u0, u1), axis=0)
            batch = np.swapaxes(batch, 0, 3)
            # batch = batch.tolist()
            batches = np.concatenate((batches, batch), axis=0)
            if len(batches) >= max_num_to_take:
                batches = batches[:max_num_to_take, ...]
                print("{} real simulated data are generated".format((len(batches))))
                return batches
        return batches

    def augmentation(self,
                     fn_im_paths,
                     motion_shares,
                     amplitude,
                     selected_frames,
                     selected_slices,
                     max_num_to_take=2000,
                     cross_test=False):
        batches = []
        # # Debug: to visualize a certain image here
        # dset = load_mat_file('../data/resp/patient/029/Ph4_Tol100_t000_Ext00_EspOff_closest_recon.mat')
        # dset = dset['dImg']
        # dset = np.array(dset, dtype=np.float32)
        # dset = np.transpose(dset, (2, 3, 1, 0))
        # pylab.imshow(dset[:, :, 51, 0])
        for fn_im_path in fn_im_paths:
            dset = load_mat_file(fn_im_path)
            dset = dset['dImg']
            try:
                dset = np.array(dset, dtype=np.float32)
                # dset = tf.constant(dset, dtype=tf.float32)
            except ImportError:
                print("File {} is defective and cannot be read!".format(fn_im_path))
                continue
            dset = np.transpose(dset, (2, 3, 1, 0))
            dset = dset[..., selected_frames]
            dset = dset[..., selected_slices, :]

            for frame in range(np.shape(dset)[3]):
                for slice in range(np.shape(dset)[2]):
                    if not cross_test:
                        img = dset[..., slice, :][..., frame]
                        img = np.flip(img, axis=0)  # todo
                        # img = (img - np.mean(img)) / np.std(img)
                        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
                        img_size = np.shape(img)
                        motion_type = np.random.choice(np.arange(0, 2), p=motion_shares)
                        u = _u_generation_2D(img_size, amplitude, motion_type=motion_type)
                        warped_img = np_warp_2D(img, u)
                        img, warped_img, = img[..., np.newaxis], warped_img[..., np.newaxis]
                        batch = np.concatenate([img, warped_img, u], 2)
                        batches.append(batch)

                        # batch = batch[np.newaxis, ...]
                        # #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        # batches = np.concatenate((batches, batch), axis=0) # this step is too slow
                        if len(batches) >= max_num_to_take:
                            batches = batches[:max_num_to_take]
                            print("{} augmented data with synthetic flows are generated".format((len(batches))))

                            return np.asarray(batches, dtype=np.float32)
                    else:
                        im1 = dset[..., slice, 0]
                        im2 = dset[..., slice, 1]
                        im1 = np.flip(im1, axis=0)  # todo
                        im2 = np.flip(im2, axis=0)  # todo
                        im1 = (im1 - np.amin(im1)) / (np.amax(im1) - np.amin(im1))
                        im2 = (im2 - np.amin(im2)) / (np.amax(im2) - np.amin(im2))
                        # im1 = (im1 - np.mean(im1)) / np.std(im1)
                        # im2 = (im2 - np.mean(im2)) / np.std(im2)
                        img_size = np.shape(im1)
                        im1, im2 = im1[..., np.newaxis], im2[..., np.newaxis]
                        u = np.zeros((*img_size, 2))
                        batch = np.concatenate([im1, im2, u], 2)
                        #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        batches.append(batch)

        return np.asarray(batches, dtype=np.float32)

    def augmentation_kspace(self,
                     fn_im_paths,
                     motion_shares,
                     amplitude,
                     selected_frames,
                     selected_slices,
                     max_num_to_take=2000,
                     cross_test=False):
        batches = []
        # # Debug: to visualize a certain image here
        # dset = load_mat_file('../data/resp/patient/029/Ph4_Tol100_t000_Ext00_EspOff_closest_recon.mat')
        # dset = dset['dImg']
        # dset = np.array(dset, dtype=np.float32)
        # dset = np.transpose(dset, (2, 3, 1, 0))
        # pylab.imshow(dset[:, :, 51, 0])
        for fn_im_path in fn_im_paths:
            dset = load_mat_file(fn_im_path)
            dset = dset['dImg']
            try:
                dset = np.array(dset, dtype=np.float32)
                # dset = tf.constant(dset, dtype=tf.float32)
            except ImportError:
                print("File {} is defective and cannot be read!".format(fn_im_path))
                continue
            dset = np.transpose(dset, (2, 3, 1, 0))
            dset = dset[..., selected_frames]
            dset = dset[..., selected_slices, :]

            for frame in range(np.shape(dset)[3]):
                for slice in range(np.shape(dset)[2]):
                    if not cross_test:
                        img = dset[..., slice, :][..., frame]
                        img = np.flip(img, axis=0)  # todo
                        # img = (img - np.mean(img)) / np.std(img)
                        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
                        img_size = np.shape(img)
                        motion_type = np.random.choice(np.arange(0, 2), p=motion_shares)
                        u = _u_generation_2D(img_size, amplitude, motion_type=motion_type)
                        warped_img = np_warp_2D(img, u)
                        img_kspace = image2kspace(img, normalize=False)
                        warped_img_kspace = image2kspace(warped_img, normalize=False)
                        img = img[..., np.newaxis]
                        warped_img = warped_img[..., np.newaxis]
                        batch = np.concatenate([img_kspace, warped_img_kspace, u, img, warped_img], 2)
                        batches.append(batch)

                        # batch = batch[np.newaxis, ...]
                        # #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        # batches = np.concatenate((batches, batch), axis=0) # this step is too slow
                        if len(batches) >= max_num_to_take:
                            batches = batches[:max_num_to_take]
                            print("{} augmented data with synthetic flows are generated".format((len(batches))))

                            return np.asarray(batches, dtype=np.float32)
        return np.asarray(batches, dtype=np.float32)


    def input_train_data(self,
                         img_dirs,
                         img_dirs_real_simulated,
                         data_per_interval,
                         selected_frames,
                         selected_slices,
                         augment_type_percent,
                         amplitude,
                         train_in_kspace,
                         crop=False,
                         cross_test=False,
                         ):

        if not train_in_kspace:
            batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
            real_simulated_data_num = math.floor(data_per_interval * augment_type_percent[2])
            if real_simulated_data_num is not 0:
                fn_im_paths = self.get_data_paths(img_dirs_real_simulated)
                np.random.shuffle(fn_im_paths)
                batches_real_simulated = self.load_real_simulated_data(fn_im_paths, selected_slices, real_simulated_data_num)
                batches = np.concatenate((batches, batches_real_simulated), axis=0)

            augmented_data_num = math.floor(data_per_interval * sum(augment_type_percent[:2]))
            if augmented_data_num is not 0:
                motion_1_share = augment_type_percent[0] / sum(augment_type_percent[:2])
                motion_2_share = augment_type_percent[1] / sum(augment_type_percent[:2])
                fn_im_paths = self.get_data_paths(img_dirs)
                np.random.shuffle(fn_im_paths)
                batches_augmented = self.augmentation(fn_im_paths,
                                                      [motion_1_share, motion_2_share],
                                                      amplitude,
                                                      selected_frames,
                                                      selected_slices,
                                                      augmented_data_num)
                batches = np.concatenate((batches, batches_augmented), axis=0)
                np.random.shuffle(batches)

            im1_queue = tf.train.slice_input_producer([batches[..., 0]], shuffle=False,
                                                      capacity=len(list(batches[..., 0])), num_epochs=None)
            im2_queue = tf.train.slice_input_producer([batches[..., 1]], shuffle=False,
                                                      capacity=len(list(batches[..., 1])), num_epochs=None)
            flow_queue = tf.train.slice_input_producer([batches[..., 2:4]], shuffle=False,
                                                       capacity=len(list(batches[..., 2:4])), num_epochs=None)
            # num_queue = tf.train.slice_input_producer([patient_num], shuffle=False,
            #                                            capacity=len(list(patient_num)), num_epochs=None)
        else:
            batches = np.zeros((0, self.dims[0], self.dims[1], 6), dtype=np.float32)
            fn_im_paths = self.get_data_paths(img_dirs)
            np.random.shuffle(fn_im_paths)
            motion_1_share = augment_type_percent[0] / sum(augment_type_percent[:2])
            motion_2_share = augment_type_percent[1] / sum(augment_type_percent[:2])
            batches_augmented = self.augmentation(fn_im_paths,
                                                  [motion_1_share, motion_2_share],
                                                  amplitude,
                                                  selected_frames,
                                                  selected_slices,
                                                  data_per_interval)
            if crop:
                batches_augmented = self.crop2D(batches_augmented, crop_size=64, box_num=8)
            batches = np.concatenate((imgpair2kspace(batches_augmented[..., :2]), batches_augmented[..., 2:]), axis=-1)

            #batches = np.concatenate((batches, batches_augmented), axis=0)
            np.random.shuffle(batches)
            flow = batches[:, 0, 0, 4:6]
            im1_queue = tf.train.slice_input_producer([batches[..., :2]], shuffle=False,
                                                      capacity=len(list(batches[..., 0])), num_epochs=None)
            im2_queue = tf.train.slice_input_producer([batches[..., 2:4]], shuffle=False,
                                                      capacity=len(list(batches[..., 1])), num_epochs=None)
            flow_queue = tf.train.slice_input_producer([flow], shuffle=False,
                                                       capacity=len(list(flow)), num_epochs=None)

        return tf.train.batch([im1_queue, im2_queue, flow_queue],
                              batch_size=self.batch_size,
                              num_threads=self.num_threads)

    def input_test_data(self,
                        test_types,
                        img_dir,
                        img_dir_matlab_simulated,
                        selected_frames,
                        selected_slices,
                        amplitude,
                        crop,
                        test_in_kspace,
                        cross_test=False):
        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        for test_type in test_types:
            if test_type is 0:
                # fn_im_paths = self.get_data_paths(img_dir)
                # batch = self.augmentation(fn_im_paths,
                #                           [1, 0],
                #                           amplitude,
                #                           selected_frames,
                #                           selected_slices)
                # batch = batch[0, ...]  # only take the first sample
                # batch = batch[np.newaxis, ...]

                fn_im_paths = self.get_data_paths(img_dir)
                batch = self.augmentation(fn_im_paths,
                                          [1, 0],
                                          amplitude,
                                          selected_frames,
                                          selected_slices)

                # batch: batch_size * im_size[0] * im_size[1] * 8,
                # 8: [fft.real &  imag of im1, fft.real &  imag of im2, u1, u2, im1, im2]

                if crop:
                    batch = self.crop2D(batch, crop_size=64, box_num=1, pos=[[100], [100]])
                #else:
                batch = np.concatenate((imgpair2kspace(batch[..., :2]), batch[..., 2:], batch[..., :2]), axis=-1)


                # batch = batch[0, ...]  # only take the first sample
                # batch = batch[np.newaxis, ...]
                im1_k_queue = tf.train.slice_input_producer([batch[..., :2]], shuffle=False,
                                                          capacity=len(list(batch[..., 0])), num_epochs=None)
                im2_k_queue = tf.train.slice_input_producer([batch[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batch[..., 1])), num_epochs=None)
                flow_queue = tf.train.slice_input_producer([batch[..., 4:6]], shuffle=False,
                                                           capacity=len(list(batch[..., 4:6])), num_epochs=None)
                im1_queue = tf.train.slice_input_producer([batch[..., 6]], shuffle=False,
                                                           capacity=len(list(batch[..., 6])), num_epochs=None)
                im2_queue = tf.train.slice_input_producer([batch[..., 7]], shuffle=False,
                                                           capacity=len(list(batch[..., 7])), num_epochs=None)
                return tf.train.batch([im1_k_queue, im2_k_queue, flow_queue, im1_queue, im2_queue],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads)
            elif test_type is 1:
                fn_im_paths = self.get_data_paths(img_dir)
                batch = self.augmentation(fn_im_paths,
                                          [0, 1],
                                          amplitude,
                                          selected_frames,
                                          selected_slices)
                batch = batch[0, ...]  # only take the first sample
                batch = batch[np.newaxis, ...]
            elif test_type is 2:
                fn_im_paths = self.get_data_paths(img_dir)
                batch = self.augmentation(fn_im_paths,
                                          [0, 1],
                                          amplitude,
                                          selected_frames,
                                          selected_slices,
                                          cross_test=True)
                batch = batch[0, ...]  # only take the first sample
                batch = batch[np.newaxis, ...]
            elif test_type is 3:
                fn_im_paths = self.get_data_paths(img_dir_matlab_simulated)
                batch = self.load_real_simulated_data(fn_im_paths, selected_slices)
            else:
                raise ImportError('Wrong test type is given')

            batches = np.concatenate((batches, batch), axis=0)

        im1_queue = tf.train.slice_input_producer([batches[..., 0]], shuffle=False,
                                                  capacity=len(list(batches[..., 0])), num_epochs=None)
        im2_queue = tf.train.slice_input_producer([batches[..., 1]], shuffle=False,
                                                  capacity=len(list(batches[..., 1])), num_epochs=None)
        flow_queue = tf.train.slice_input_producer([batches[..., 2:4]], shuffle=False,
                                                   capacity=len(list(batches[..., 2:4])), num_epochs=None)
        # num_queue = tf.train.slice_input_producer([patient_num], shuffle=False,
        #                                            capacity=len(list(patient_num)), num_epochs=None)
        return tf.train.batch([im1_queue, im2_queue, flow_queue],
                              batch_size=self.batch_size,
                              num_threads=self.num_threads)

    def input_train_gt(self):
        img_dirs = ['resp/patient',
                    'resp/volunteer']
        selected_frames = [0, 3]
        selected_slices = list(range(15, 55))
        amplitude = 30
        flow_augment_type = ['constant', 'smooth']

        fn_im_paths = self.get_data_paths(img_dirs)

        data_info_list = []
        # length_data = len(fn_im_paths) * len(selected_frames) * len(selected_slices)

        for fn_im_path in fn_im_paths:
            for frame in selected_frames:
                for slice in selected_slices:
                    # data_info_list.append([fn_im_path, frame, slice, random.choice(flow_augment_type)])
                    data_info_list.append(fn_im_path + ',' + str(frame) + ','
                                          + str(slice) + ',' + random.choice(flow_augment_type))

        random.seed(0)
        random.shuffle(data_info_list)

        dataset = tf.data.TFRecordDataset(data_info_list)

        dataset = dataset.map(map_func=lambda file_path: self.load_batch(self.convert2list(file_path)), num_parallel_calls=True)

        return tf.train.batch(
            dataset,
            batch_size=self.batch_size,
            num_threads=self.num_threads)

    def convert2list(self, file_path):
        file_element = file_path.split(',')
        return file_element[0], int(file_element[1]), int(file_element[2]), file_element[3]

    def load_batch(self, fn_im_path, frame, slice, flow_augment_type):
        with h5py.File(fn_im_path, 'r') as f:
            # fn_im_raw = sio.loadmat(fn_im_path)
            dset = f['dImg']
            try:
                dset = np.array(dset, dtype=np.float32)
                # dset = tf.constant(dset, dtype=tf.float32)
            except Exception:
                print("File {} is defective and cannot be read!".format(fn_im_path))
            dset = np.transpose(dset, (2, 3, 1, 0))
            img = dset[..., slice, :][..., frame]
            img_size = np.shape(img)
            u = self._u_generation_2D(img_size, amplitude, motion_type=flow_augment_type)
            warped_img = self._np_warp_2D(img, u)
            img, warped_img, = img[..., np.newaxis], warped_img[..., np.newaxis]
            mask = np.zeros(img_size)
        return [img, warped_img, u, mask]


class MRI_Resp_3D(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def u_generation_3D(self, img_size, amplitude, motion_type='constant'):
        [M, N, P] = img_size
        if motion_type == 'constant':
            u_C = np.random.rand(3)
            amplitude = amplitude / np.linalg.norm(u_C, 2)
            u = amplitude * np.ones((M, N, P, 3))
            u[..., 0] = u_C[0] * u[..., 0]
            u[..., 1] = u_C[1] * u[..., 1]
            u[..., 2] = u_C[2] * u[..., 2]
        elif motion_type == 'smooth':

            u = np.random.rand(M, N, P, 3)

            cut_off = 0.01
            w_x_cut = math.floor(cut_off / (1 / M) + (M + 1) / 2)
            w_y_cut = math.floor(cut_off / (1 / N) + (N + 1) / 2)
            w_z_cut = math.floor(cut_off / (1 / P) + (P + 1) / 2)

            LowPass_win = np.zeros((M, N, P))
            LowPass_win[(M - w_x_cut): w_x_cut, (N - w_y_cut): w_y_cut, (P - w_z_cut): w_z_cut] = 1

            u[..., 0] = (np.fft.ifftn(np.fft.fftn(u[..., 0]) * np.fft.ifftshift(LowPass_win))).real
            u[..., 1] = (np.fft.ifftn(np.fft.fftn(u[..., 1]) * np.fft.ifftshift(LowPass_win))).real
            u[..., 2] = (np.fft.ifftn(np.fft.fftn(u[..., 2]) * np.fft.ifftshift(LowPass_win))).real         

        elif motion_type == 'realistic':
            pass

        return u

    def warp_3D(self, img, flow):
        img = img.astype('float32')
        flow = flow.astype('float32')
        height, width, depth = self.dims
        posx, posy, posz = np.mgrid[:height, :width, :depth]
        # flow=np.reshape(flow, [-1, 3])
        vx = flow[:, :, 0]
        vy = flow[:, :, 1]
        vz = flow[:, :, 2]

        coord_x = posx + vx
        coord_y = posy + vy
        coord_z = posz + vz
        coords = np.array([coord_x, coord_y, coord_z])
        warped = warp(img, coords, order=1)  # order=1 for bi-linear

        return warped

    def input_train_gt(self, hold_out):
        img_dirs = ['resp/patient',
                    'resp/volunteer']
        selected_frames = [0, 3]
        amplitude = 30

        height, width = self.dims
        filenames = []
        dataset_filenames = []
        for img_dir in img_dirs:
            img_dir = os.path.join(self.data.current_dir, img_dir)
            img_files = os.listdir(img_dir)
            for img_file in img_files:
                if not img_file.startswith('.'):
                    try:
                        img_mat = os.listdir(os.path.join(img_dir, img_file))[0]

                        fn_im_path = os.path.join(img_dir, img_file, img_mat)
                        with h5py.File(fn_im_path, 'r') as f:
                            # fn_im_raw = sio.loadmat(fn_im_path)
                            dset = f['dImg']
                            dset = np.array(dset, dtype=np.float32)
                            dset = np.transpose(dset, (2, 3, 1, 0))
                            dset = dset[..., selected_frames]
                        for frame in range(np.shape(dset)[3]):
                            img_size = np.shape(dset[..., frame])
                            u = self.u_generation_3D(img_size, amplitude, motion_type='smooth')

                            pass



                        dataset_filenames.append(fn_im_path)
                    except Exception:
                        print("File {} is empty!".format(img_file))
                        pass

        for img_dir, gt_dir in zip(img_dirs, gt_dirs):
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            img_files.sort()
            gt_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                fn_im1 = os.path.join(img_dir, img_files[2 * i])
                fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                dataset_filenames.append((fn_im1, fn_im2, fn_gt))

            random.seed(0)
            random.shuffle(dataset_filenames)
            dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        random.seed(0)
        random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        fns_im1, fns_im2, fns_gt = zip(*filenames)
        fns_im1 = list(fns_im1)
        fns_im2 = list(fns_im2)
        fns_gt = list(fns_gt)

        im1 = read_png_image(fns_im1)
        im2 = read_png_image(fns_im2)
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)

        im1, im2, gt = random_crop([im1, im2, gt],
                                   [height, width, 3])
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im1 = self._normalize_image(im1)
            im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im1, im2, flow_gt, mask_gt],
            batch_size=self.batch_size,
            num_threads=self.num_threads)


class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _preprocess_flow(self, gt):
        flow, mask = gt
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        flow = tf.reshape(self._resize_crop_or_pad(flow), [height, width, 2])
        mask = tf.reshape(self._resize_crop_or_pad(mask), [height, width, 1])
        return flow, mask

    def _input_flow(self, flow_dir, hold_out_inv):
        flow_dir_occ = os.path.join(self.data.current_dir, flow_dir, 'flow_occ')
        flow_dir_noc = os.path.join(self.data.current_dir, flow_dir, 'flow_noc')
        filenames_gt_occ = []
        filenames_gt_noc = []
        flow_files_occ = os.listdir(flow_dir_occ)
        flow_files_occ.sort()
        flow_files_noc = os.listdir(flow_dir_noc)
        flow_files_noc.sort()

        if hold_out_inv is not None:
            random.seed(0)
            random.shuffle(flow_files_noc)
            flow_files_noc = flow_files_noc[:hold_out_inv]

            random.seed(0)
            random.shuffle(flow_files_occ)
            flow_files_occ = flow_files_occ[:hold_out_inv]

        assert len(flow_files_noc) == len(flow_files_occ)

        for i in range(len(flow_files_occ)):
            filenames_gt_occ.append(os.path.join(flow_dir_occ,
                                                 flow_files_occ[i]))
            filenames_gt_noc.append(os.path.join(flow_dir_noc,
                                                 flow_files_noc[i]))

        flow_occ, mask_occ = self._preprocess_flow(
            _read_flow(filenames_gt_occ, 1))
        flow_noc, mask_noc = self._preprocess_flow(
            _read_flow(filenames_gt_noc, 1))
        return flow_occ, mask_occ, flow_noc, mask_noc

    def _input_train(self, image_dir, flow_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        flow_occ, mask_occ, flow_noc, mask_noc = self._input_flow(flow_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape, flow_occ, mask_occ, flow_noc, mask_noc],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_train_gt(self, hold_out):
        img_dirs = ['data_scene_flow/training/image_2',
                    'data_stereo_flow/training/colored_0']
        gt_dirs = ['data_scene_flow/training/flow_occ',
                   'data_stereo_flow/training/flow_occ']

        height, width = self.dims

        filenames = []
        for img_dir, gt_dir in zip(img_dirs, gt_dirs):
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            img_files.sort()
            gt_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                fn_im1 = os.path.join(img_dir, img_files[2 * i])
                fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                dataset_filenames.append((fn_im1, fn_im2, fn_gt))

            random.seed(0)
            random.shuffle(dataset_filenames)
            dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        random.seed(0)
        random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        fns_im1, fns_im2, fns_gt = zip(*filenames)
        fns_im1 = list(fns_im1)
        fns_im2 = list(fns_im2)
        fns_gt = list(fns_gt)

        im1 = read_png_image(fns_im1)
        im2 = read_png_image(fns_im2)
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)

        im1, im2, gt = random_crop([im1, im2, gt],
                                   [height, width, 3])
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im1 = self._normalize_image(im1)
            im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im1, im2, flow_gt, mask_gt],
            batch_size=self.batch_size,
            num_threads=self.num_threads)

    def input_train_2015(self, hold_out_inv=None):
        return self._input_train('data_scene_flow/training/image_2',
                                 'data_scene_flow/training',
                                 hold_out_inv)

    def input_test_2015(self, hold_out_inv=None):
        return self._input_test('data_scene_flow/testing/image_2', hold_out_inv)

    def input_train_2012(self, hold_out_inv=None):
        return self._input_train('data_stereo_flow/training/colored_0',
                                 'data_stereo_flow/training',
                                 hold_out_inv)

    def input_test_2012(self, hold_out_inv=None):
        return self._input_test('data_stereo_flow/testing/colored_0', hold_out_inv)
