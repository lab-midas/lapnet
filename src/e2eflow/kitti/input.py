import os
import sys
import math

import time
import cProfile


import numpy as np
import tensorflow as tf
import random
from skimage.util.shape import view_as_windows
import pylab
from multiprocessing import Pool
import matplotlib.pyplot as plt

from ..core.input import read_png_image, Input, load_mat_file
from ..core.augment import random_crop
from ..core.flow_util import flow_to_color
from ..core.image_warp import np_warp_2D, np_warp_3D
from ..core.sampling import generate_mask
from ..core.sampling_2 import generate_mask_2


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


def pos_generation_2D(intervall, stride):
    """

    :param intervall:
    :param stride:
    :return: 2 x position (x, y)
    """
    x = np.arange(intervall[0][0], intervall[0][1], stride)
    y = np.arange(intervall[1][0], intervall[1][1], stride)
    vx, vy = np.meshgrid(x, y)
    vx = vx.reshape(vx.shape[1] * vx.shape[0])
    vy = vy.reshape(vy.shape[1] * vy.shape[0])
    pos = np.stack((vx, vy), axis=0)
    return pos


def visualise(patient_path, frame, slice):
    """
    to visualize a certain image
    :param patient_path:
    :param frame:
    :param slice:
    :return:
    """
    # dset = load_mat_file('../data/resp/patient/029/Ph4_Tol100_t000_Ext00_EspOff_closest_recon.mat')
    dset = load_mat_file(patient_path)
    dset = dset['dImg']
    dset = np.array(dset, dtype=np.float32)
    dset = np.transpose(dset, (3, 2, 1, 0))
    pylab.imshow(np.rot90(dset[:, :, slice, frame]))


def _u_generation_3D(img_size, amplitude, motion_type=0):

    M, N, P = img_size
    if motion_type == 0:
        # amplitude = np.random.randint(0, amplitude)
        u_C = -1 + 2 * np.random.rand(3)  # interval [-1, 1]
        u_C[2] = 0  # todo
        amplitude = amplitude / np.linalg.norm(u_C, 2)
        u = amplitude * np.ones((M, N, P, 3))
        u[..., 0] = u_C[0] * u[..., 0]
        u[..., 1] = u_C[1] * u[..., 1]
        u[..., 2] = u_C[2] * u[..., 2]
    elif motion_type == 1:
        u = np.random.normal(0, 1, (M, N, P, 3))
        u[..., 2] = 0  # todo
        cut_off = 0.01
        w_x_cut = math.floor(cut_off / (1 / M) + (M + 1) / 2)
        w_y_cut = math.floor(cut_off / (1 / N) + (N + 1) / 2)
        w_z_cut = math.floor(cut_off / (1 / P) + (P + 1) / 2)

        LowPass_win = np.zeros((M, N, P), dtype=np.float32)
        LowPass_win[(M - w_x_cut): w_x_cut, (N - w_y_cut): w_y_cut, (P - w_z_cut):w_z_cut] = 1

        u[..., 0] = (np.fft.ifftn(np.fft.fftn(u[..., 0]) * np.fft.ifftshift(LowPass_win))).real
        u[..., 1] = (np.fft.ifftn(np.fft.fftn(u[..., 1]) * np.fft.ifftshift(LowPass_win))).real
        u[..., 2] = (np.fft.ifftn(np.fft.fftn(u[..., 2]) * np.fft.ifftshift(LowPass_win))).real

        u1 = u[..., 0].flatten()
        u2 = u[..., 1].flatten()
        u3 = u[..., 2].flatten()
        amplitude = amplitude / max(np.linalg.norm(np.vstack([u1, u2, u3]), axis=0))
        u = u * amplitude

    return u


def _u_generation_2D(img_size, amplitude, motion_type=0):
    """

    :param img_size:
    :param amplitude:
    :param motion_type: 0: constant, 1: smooth
    :return:
    """
    M, N = img_size
    # amplitude = np.random.randint(0, amplitude_in)
    if motion_type == 0:
        amplitude = np.random.randint(0, amplitude)
        #u_C = 2 * np.random.rand(2)
        u_C = -1 + 2 * np.random.rand(2)  # interval [-1, 1]
        amplitude = amplitude / np.linalg.norm(u_C, 2)
        u = amplitude * np.ones((M, N, 2))
        u[..., 0] = u_C[0] * u[..., 0]
        u[..., 1] = u_C[1] * u[..., 1]
        pass
    elif motion_type == 1:
        u = -1 + 2*np.random.rand(M, N, 2)
        # u = np.random.normal(0, 1, (M, N, 2))
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

# # deprecated
# def imgpair2kspace(arr, normalize=False):
#     im1_kspace = to_freq_space(arr[..., 0])
#     im2_kspace = to_freq_space(arr[..., 1])
#
#     return np.asarray(np.concatenate((im1_kspace, im2_kspace), axis=-1), dtype=np.float32)


def arr2kspace(arr, normalize=False):
    """
    convert a 4D array (batch_size, x_dim, y_dim, channels) to kspace along the last axis, FFT on x and y dimension
    :param arr:
    :param normalize:
    :return: (batch_size, x_dim, y_dim, 2 * channel)
    """
    if arr.dtype == np.float64:
        arr = arr.astype(dtype=np.float32)
    arr_kspace = np.zeros((np.shape(arr)[0], np.shape(arr)[1], np.shape(arr)[2], 2*np.shape(arr)[3]), dtype=np.float32)
    for i in range(np.shape(arr)[-1]):
        kspace = to_freq_space(arr[..., i], normalize=normalize)
        arr_kspace[..., 2*i:2*i+2] = kspace
    return arr_kspace


# #deprecated
# def image2kspace(y, normalize=False):
#     """
#     Prepares frequency data from image data: applies to_freq_space,
#     expands the dimensions from 3D to 4D, and normalizes if normalize=True
#     :param y: input image
#     :param normalize: if True - the frequency data will be normalized
#     :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
#     """
#     x = to_freq_space(y)  # FFT: (256, 256, 2)
#     #x = np.expand_dims(x, axis=0)  # (1, 256, 256, 2)
#     if normalize:
#         x = x - np.mean(x)
#
#     return x


# haven't be tested
def to_freq_space_tf(img):
    img_f = tf.signal.fft2d(img)  # FFT
    img_fshift = tf.signal.fftshift(img_f, axes=(-1, -2))  # FFT shift todo: test this shift!
    # sess = tf.InteractiveSession()
    # a = img_fshift.eval()
    # b = img_f.eval()
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def to_freq_space(img, normalize=False):
    """ Performs FFT of images
    :param img: input n (batches) x 2D images
    :return: n (batches) Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f, axes=(-1, -2))  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.stack((img_real, img_imag), axis=-1)  # (im_size1, im_size2, 2)
    if normalize:
        img_real_imag = (img_real_imag.transpose() - np.mean(img_real_imag, axis=(1, 2, 3))).transpose()
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
            if os.path.isfile(img_dir):
                fn_im_paths.append(img_dir)
                continue
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


    def crop2D_FixPts(self, arr, crop_size, box_num, pos):
        """

        :param arr: shape:[batch_size, img_size, img_size, n]
        :param crop_size:
        :param box_num:
        :param pos:
        :return:
        """
        arr_cropped_augmented = np.zeros((np.shape(arr)[0] * box_num, crop_size, crop_size, np.shape(arr)[-1]),
                                         dtype=np.float32)
        # if len(arr.shape()) is 4:
        #     arr = arr[0, ...]
        # elif len(arr.shape()) is 2:
        #     arr = arr[..., np.newaxis]
        # elif len(arr.shape()) is 3:
        #     pass
        # else:
        #     raise ImportError
        for batch in range(np.shape(arr)[0]):
            for i in range(box_num):
                arr_cropped_augmented[batch * box_num + i, ...] = arr[batch,
                                                                      pos[0][i]:pos[0][i] + crop_size,
                                                                      pos[1][i]:pos[1][i] + crop_size,
                                                                      :]

        return arr_cropped_augmented

    def crop2D(self, arr, crop_size, box_num, pos=None):
        """
        :param arr:
        :param crop_size:
        :param box_num: crops per slices
        :param pos: shape (2, n), pos_x and pos_y, n must be same as the box number. if pos=None, select random pos
        :return:
        """
        arr_cropped_augmented = np.zeros((arr.shape[0] * box_num, crop_size, crop_size, arr.shape[-1]), dtype=np.float32)
        x_dim, y_dim = np.shape(arr)[1:3]
        for i in range(box_num):
            if pos is None:
                x_pos = np.random.randint(0, x_dim - crop_size + 1, arr.shape[0])
                y_pos = np.random.randint(0, y_dim - crop_size + 1, arr.shape[0])
            else:
                x_pos = pos[0]
                y_pos = pos[1]

            w = view_as_windows(arr, (1, crop_size, crop_size, 1))[..., 0, :, :, 0]
            out = w[np.arange(arr.shape[0]), x_pos, y_pos]
            out = out.transpose(0, 2, 3, 1)
            arr_cropped_augmented[i * arr.shape[0]:(i + 1) * arr.shape[0], ...] = out
        return arr_cropped_augmented

    def load_real_simulated_data(self, fn_im_paths, selected_slices, max_num_to_take=2000):
        batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
        for fn_im_path in fn_im_paths:
            f = load_mat_file(fn_im_path)
            if np.shape(f['I1_real'])[2] != 72:
                continue
            ref = f['I1_real']
            mov = f['I1_Real_hat']

            # mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc30.npy')
            acc = np.random.choice(np.arange(1, 32, 6))
            mask = np.transpose(generate_mask(nSegments=25, acc=acc, nRep=4), (2, 1, 0))
            # mask_2 = np.transpose(generate_mask_2(subsampleType=1, acc=4, vd_type=1, nRep=4), (2, 1, 0))
            k_dset = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
            k_warped_dset = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
            ref = (np.fft.ifftn(k_dset)).real
            mov = (np.fft.ifftn(k_warped_dset)).real

            u0 = -f['u_Real_est_1']
            u1 = -f['u_Real_est_2']
            im1 = ref[..., selected_slices]
            im1 = im1[np.newaxis, ...]
            im2 = mov[..., selected_slices]
            im2 = im2[np.newaxis, ...]
            u0 = u0[..., selected_slices]
            u0 = u0[np.newaxis, ...]
            u1 = u1[..., selected_slices]
            u1 = u1[np.newaxis, ...]

            # im1 = np.squeeze(im1[..., 0])
            # im2 = np.squeeze(im2[..., 0])
            # im1 = (im1 - np.amin(im1)) / (np.amax(im1) - np.amin(im1))
            # im2 = (im2 - np.amin(im2)) / (np.amax(im2) - np.amin(im2))
            # u0 = u0[..., 0]
            # u1 = u1[..., 0]
            # u = np.concatenate((u0, u1), axis=0)
            # u = np.swapaxes(u, 0, 2)
            # u_swap = np.swapaxes(u, 0, 1)
            # pass
            # im1_hat = np_warp_2D(im2, -u)
            # im1_hat_2 = np_warp_2D(im2, -u_swap)  # THIS ONE IS RIGHT!!!
            # im1_hat_3 = np_warp_2D(im2, u)
            # im1_hat_4 = np_warp_2D(im2, u_swap)
            # # im1_hat_hat = np_warp_2D(im2, u)
            # ori_error = im1 - im2
            #
            # warped_error_1 = im1 - im1_hat
            # warped_error_2 = im1 - im1_hat_2
            # warped_error_3 = im1 - im1_hat_3
            # warped_error_4 = im1 - im1_hat_4
            # fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            # ax[0][0].imshow(ori_error, cmap='gray')  # ref
            # ax[0][1].imshow(warped_error_1, cmap='gray')  # mov
            # ax[0][2].imshow(warped_error_2, cmap='gray')
            # ax[1][1].imshow(warped_error_3, cmap='gray')  # mov
            # ax[1][2].imshow(warped_error_4, cmap='gray')


            # ax[2].imshow(warped_error_2, cmap='gray')  # mov

            batch = np.concatenate((im1, im2, u0, u1), axis=0)
            batch = np.swapaxes(batch, 0, 3)
            # batch = batch.tolist()
            batches = np.concatenate((batches, batch), axis=0)
            if len(batches) >= max_num_to_take:
                batches = batches[:max_num_to_take, ...]
                print("{} real simulated data are generated".format((len(batches))))
                return np.asarray(batches, dtype=np.float32)
        return np.asarray(batches, dtype=np.float32)

    def augmentation_3D(self,
                        fn_im_paths,
                        motion_shares,
                        amplitude,
                        selected_frames,
                        selected_slices,
                        given_u=False,
                        max_num_to_take=4000,
                        undersampling=True,
                        cross_test=False
                        ):

        batches = np.zeros((0, self.dims[0], self.dims[1], 6), dtype=np.float32)
        for fn_im_path in fn_im_paths:
            dset_orig = load_mat_file(fn_im_path)
            dset_orig = dset_orig['dImg']
            # dset_orig = dset_orig['dMBCSHDPROST']
            try:
                dset_orig = np.array(dset_orig, dtype=np.float32)
                # dset = tf.constant(dset, dtype=tf.float32)
            except ImportError:
                print("File {} is defective and cannot be read!".format(fn_im_path))
                continue
            if not cross_test:
                for frame in selected_frames:
                    dset = np.transpose(dset_orig, (3, 2, 1, 0))
                    dset = (dset - np.amin(dset)) / (np.amax(dset) - np.amin(dset))
                    #dset = (dset - np.mean(dset)) / np.std(dset)
                    dset = dset[..., frame]
                    dset = np.rot90(dset, axes=(0, 1))
                    img_size = np.shape(dset)
                    if img_size[2] is not 72:
                        continue
                    if not given_u:
                        motion_type = np.random.choice(np.arange(0, 2), p=motion_shares)
                        u = _u_generation_3D(img_size, amplitude, motion_type=motion_type)  # TODO: This step too time-consuming
                        #np.save('/home/jpa19/PycharmProjects/MA/UnFlow/new_u_amp10_3D', u)
                    else:
                        u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/u_smooth_apt10_3D.npy')
                    warped_dset = np_warp_3D(dset, u)
                    acc = np.random.choice(np.arange(1, 32, 6))
                    mask = np.transpose(generate_mask(nSegments=25, acc=acc, nRep=4), (2, 1, 0))
                    # mask_2 = np.transpose(generate_mask_2(subsampleType=1, acc=acc, vd_type=1, nRep=4), (2, 1, 0))
                    # mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc30.npy')
                    k_dset = np.multiply(np.fft.fftn(dset), np.fft.ifftshift(mask[0, ...]))
                    k_warped_dset = np.multiply(np.fft.fftn(warped_dset), np.fft.ifftshift(mask[3, ...]))
                    dset_us = (np.fft.ifftn(k_dset)).real
                    warped_dset_us = (np.fft.ifftn(k_warped_dset)).real

                    # k_dset_2 = np.multiply(np.fft.fftn(dset), np.fft.ifftshift(mask_2[0, ...]))
                    # k_warped_dset_2 = np.multiply(np.fft.fftn(warped_dset), np.fft.ifftshift(mask_2[3, ...]))
                    # dset_us_2 = (np.fft.ifftn(k_dset_2)).real
                    # warped_dset_us_2 = (np.fft.ifftn(k_warped_dset_2)).real
                    # fig, ax = plt.subplots(3, 2, figsize=(6, 10))
                    # ax[0][0].imshow(dset[..., selected_slices[0]])  # ref
                    # ax[0][0].set_title('Ref Img')
                    # ax[0][1].imshow(warped_dset[..., selected_slices[0]])  # mov
                    # ax[0][1].set_title('Moving Img')
                    # ax[1][0].imshow(dset_us[..., selected_slices[0]])
                    # ax[1][0].set_title('Ref US 1 Img')
                    # ax[1][1].imshow(warped_dset_us[..., selected_slices[0]])
                    # ax[1][1].set_title('Moving US 1 Img')
                    # ax[2][0].imshow(dset_us_2[..., selected_slices[0]])
                    # ax[2][0].set_title('Ref US 2 Img')
                    # ax[2][1].imshow(warped_dset_us_2[..., selected_slices[0]])
                    # ax[2][1].set_title('Moving US 2 Img')
                    # plt.show()
                    # pass

                    dset_us, warped_dset_us = dset_us[..., np.newaxis], warped_dset_us[..., np.newaxis]
                    dset, warped_dset = dset[..., np.newaxis], warped_dset[..., np.newaxis]

                    batch = np.concatenate((dset_us, warped_dset_us, u[..., :2], dset, warped_dset), axis=-1)
                    batch = np.transpose(batch, (2, 0, 1, 3))
                    batch = batch[selected_slices, ...]
                    batches = np.concatenate((batches, batch), axis=0)
                    if len(batches) >= max_num_to_take:
                        batches = batches[:max_num_to_take, ...]
                        print("{} augmented data with synthetic flows are generated".format((len(batches))))
                        return np.asarray(batches, dtype=np.float32)
            else:
                dset_orig = np.transpose(dset_orig, (3, 2, 1, 0))
                dset_orig = (dset_orig - np.amin(dset_orig)) / (np.amax(dset_orig) - np.amin(dset_orig))
                dset_orig = dset_orig[..., selected_frames]
                dset = dset_orig[..., 0]
                warped_dset = dset_orig[..., 1]
                dset = np.rot90(dset, axes=(0, 1))
                warped_dset = np.rot90(warped_dset, axes=(0, 1))
                img_size = np.shape(dset)
                mask = np.transpose(generate_mask(nSegments=25, acc=30), (2, 1, 0))
                k_dset = np.multiply(np.fft.fftn(dset), np.fft.ifftshift(mask))
                k_warped_dset = np.multiply(np.fft.fftn(warped_dset), np.fft.ifftshift(mask))
                dset_us = (np.fft.ifftn(k_dset)).real
                warped_dset_us = (np.fft.ifftn(k_warped_dset)).real
                dset_us, warped_dset_us = dset_us[..., np.newaxis], warped_dset_us[..., np.newaxis]
                dset, warped_dset = dset[..., np.newaxis], warped_dset[..., np.newaxis]
                u = np.zeros((*img_size, 2))
                batch = np.concatenate((dset_us, warped_dset_us, u[..., :2], dset, warped_dset), axis=-1)
                batch = np.transpose(batch, (2, 0, 1, 3))
                batch = batch[selected_slices, ...]
                batches = np.concatenate((batches, batch), axis=0)
                if len(batches) >= max_num_to_take:
                    batches = batches[:max_num_to_take, ...]
                    return np.asarray(batches, dtype=np.float32)

        return np.asarray(batches, dtype=np.float32)

    def augmentation(self,
                     fn_im_paths,
                     motion_shares,
                     amplitude,
                     selected_frames,
                     selected_slices,
                     given_u=False,
                     max_num_to_take=4000,
                     cross_test=False):
        batches = []
        for fn_im_path in fn_im_paths:
            dset = load_mat_file(fn_im_path)
            dset = dset['dImg']
            try:
                dset = np.array(dset, dtype=np.float32)
                # dset = tf.constant(dset, dtype=tf.float32)
            except ImportError:
                print("File {} is defective and cannot be read!".format(fn_im_path))
                continue
            dset = np.transpose(dset, (3, 2, 1, 0))
            if np.shape(dset)[2] is not 72:
                continue
            dset = dset[..., selected_frames]
            dset = dset[..., selected_slices, :]

            for frame in range(np.shape(dset)[3]):
                for slice in range(np.shape(dset)[2]):
                    if not cross_test:
                        img = dset[..., slice, :][..., frame]
                        # img = np.flip(img, axis=0)
                        img = np.rot90(img)
                        #img = (img - np.mean(img)) / np.std(img)
                        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
                        img_size = np.shape(img)
                        motion_type = np.random.choice(np.arange(0, 2), p=motion_shares)
                        if not given_u:
                            u = _u_generation_2D(img_size, amplitude, motion_type=motion_type)
                        else:
                            u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/u_amp10_3D.npy')
                            u = u[:, :, selected_slices[0], :2]
                        warped_img = np_warp_2D(img, u)

                        # # show the warp error
                        # img_hat = np_warp_2D(warped_img, -u)
                        # ori_error = img - warped_img
                        # warped_error_1 = img - img_hat
                        # ori_error = (ori_error - np.amin(ori_error)) / (np.amax(ori_error) - np.amin(ori_error))
                        # warped_error_1 = (warped_error_1 - np.amin(warped_error_1)) / (
                        #             np.amax(warped_error_1) - np.amin(warped_error_1))
                        # fig, ax = plt.subplots(1, 2, figsize=(15, 8))
                        # ax[0].imshow(ori_error, cmap='gray')  # ref
                        # ax[1].imshow(warped_error_1, cmap='gray')  # mov

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
                        dset = (dset - np.amin(dset)) / (np.amax(dset) - np.amin(dset))
                        im1 = dset[..., slice, 0]
                        im2 = dset[..., slice, 1]
                        im1 = np.rot90(im1)
                        im2 = np.rot90(im2)
                        # im1 = (im1 - np.amin(im1)) / (np.amax(im1) - np.amin(im1))
                        # im2 = (im2 - np.amin(im2)) / (np.amax(im2) - np.amin(im2))
                        # im1 = (im1 - np.mean(im1)) / np.std(im1)
                        # im2 = (im2 - np.mean(im2)) / np.std(im2)
                        img_size = np.shape(im1)
                        im1, im2 = im1[..., np.newaxis], im2[..., np.newaxis]
                        u = np.zeros((*img_size, 2))
                        batch = np.concatenate([im1, im2, u], 2)
                        #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
                        batches.append(batch)
                        return np.asarray(batches, dtype=np.float32)

        return np.asarray(batches, dtype=np.float32)

    # #deprecated
    # def augmentation_kspace(self,
    #                  fn_im_paths,
    #                  motion_shares,
    #                  amplitude,
    #                  selected_frames,
    #                  selected_slices,
    #                  max_num_to_take=2000,
    #                  cross_test=False):
    #     batches = []
    #     # # Debug: to visualize a certain image here
    #     # dset = load_mat_file('../data/resp/patient/029/Ph4_Tol100_t000_Ext00_EspOff_closest_recon.mat')
    #     # dset = dset['dImg']
    #     # dset = np.array(dset, dtype=np.float32)
    #     # dset = np.transpose(dset, (2, 3, 1, 0))
    #     # pylab.imshow(dset[:, :, 51, 0])
    #     for fn_im_path in fn_im_paths:
    #         dset = load_mat_file(fn_im_path)
    #         dset = dset['dImg']
    #         try:
    #             dset = np.array(dset, dtype=np.float32)
    #             # dset = tf.constant(dset, dtype=tf.float32)
    #         except ImportError:
    #             print("File {} is defective and cannot be read!".format(fn_im_path))
    #             continue
    #         dset = np.transpose(dset, (2, 3, 1, 0))
    #         dset = dset[..., selected_frames]
    #         dset = dset[..., selected_slices, :]
    #
    #         for frame in range(np.shape(dset)[3]):
    #             for slice in range(np.shape(dset)[2]):
    #                 if not cross_test:
    #                     img = dset[..., slice, :][..., frame]
    #                     img = np.flip(img, axis=0)  # todo
    #                     # img = (img - np.mean(img)) / np.std(img)
    #                     img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    #                     img_size = np.shape(img)
    #                     motion_type = np.random.choice(np.arange(0, 2), p=motion_shares)
    #                     u = _u_generation_2D(img_size, amplitude, motion_type=motion_type)
    #                     warped_img = np_warp_2D(img, u)
    #                     img_kspace = image2kspace(img, normalize=False)
    #                     warped_img_kspace = image2kspace(warped_img, normalize=False)
    #                     img = img[..., np.newaxis]
    #                     warped_img = warped_img[..., np.newaxis]
    #                     batch = np.concatenate([img_kspace, warped_img_kspace, u, img, warped_img], 2)
    #                     batches.append(batch)
    #
    #                     # batch = batch[np.newaxis, ...]
    #                     # #  batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    #                     # batches = np.concatenate((batches, batch), axis=0) # this step is too slow
    #                     if len(batches) >= max_num_to_take:
    #                         batches = batches[:max_num_to_take]
    #                         print("{} augmented data with synthetic flows are generated".format((len(batches))))
    #
    #                         return np.asarray(batches, dtype=np.float32)
    #     return np.asarray(batches, dtype=np.float32)

    def input_train_data(self,
                         img_dirs,
                         img_dirs_real_simulated,
                         selected_frames,
                         params):

        middle_slices = 40
        slices_num_half = int(params.get('slices_to_take') / 2)
        selected_slices = list(range(middle_slices - slices_num_half, middle_slices + slices_num_half))

        if params.get('network') == 'flownet':
            batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
            real_simulated_data_num = math.floor(params.get('data_per_interval') * params.get('augment_type_percent')[2])
            if real_simulated_data_num is not 0:
                fn_im_paths = self.get_data_paths(img_dirs_real_simulated)
                np.random.shuffle(fn_im_paths)
                batches_real_simulated = self.load_real_simulated_data(fn_im_paths, selected_slices, real_simulated_data_num)
                batches = np.concatenate((batches, batches_real_simulated), axis=0)

            augmented_data_num = math.floor(params.get('data_per_interval') * sum(params.get('augment_type_percent')[:2]))
            if augmented_data_num is not 0:
                motion_1_share = params.get('augment_type_percent')[0] / sum(params.get('augment_type_percent')[:2])
                motion_2_share = params.get('augment_type_percent')[1] / sum(params.get('augment_type_percent')[:2])
                fn_im_paths = self.get_data_paths(img_dirs)
                np.random.shuffle(fn_im_paths)
                batches_augmented = self.augmentation_3D(fn_im_paths,
                                                      [motion_1_share, motion_2_share],
                                                      params.get('flow_amplitude'),
                                                      selected_frames,
                                                      selected_slices,
                                                      given_u=False,
                                                      max_num_to_take=augmented_data_num)
                batches_augmented = batches_augmented[..., :4]  # no undersampling
                batches = np.concatenate((batches, batches_augmented), axis=0)
                # batches = np.concatenate((arr2kspace(batches[..., :2]), batches[..., 2:]), axis=-1)
                np.random.shuffle(batches)

            im1_queue = tf.train.slice_input_producer([batches[..., 0]], shuffle=False,
                                                      capacity=len(list(batches[..., 0])), num_epochs=None)
            im2_queue = tf.train.slice_input_producer([batches[..., 1]], shuffle=False,
                                                      capacity=len(list(batches[..., 1])), num_epochs=None)
            flow_queue = tf.train.slice_input_producer([batches[..., 2:]], shuffle=False,
                                                       capacity=len(list(batches[..., 2:4])), num_epochs=None)
            # num_queue = tf.train.slice_input_producer([patient_num], shuffle=False,
            #                                            capacity=len(list(patient_num)), num_epochs=None)
            return tf.train.batch([im1_queue, im2_queue, flow_queue],
                                  batch_size=self.batch_size,
                                  num_threads=self.num_threads)
        elif params.get('network') == 'ftflownet':
            batches = np.zeros((0, self.dims[0], self.dims[1], 4), dtype=np.float32)
            real_simulated_data_num = math.floor(
                params.get('data_per_interval') * params.get('augment_type_percent')[2])
            if real_simulated_data_num is not 0:
                fn_im_paths = self.get_data_paths(img_dirs_real_simulated)
                np.random.shuffle(fn_im_paths)
                batches_real_simulated = self.load_real_simulated_data(fn_im_paths, selected_slices,
                                                                       real_simulated_data_num)
                batches = np.concatenate((batches, batches_real_simulated), axis=0)

            augmented_data_num = math.floor(
                params.get('data_per_interval') * sum(params.get('augment_type_percent')[:2]))
            if augmented_data_num is not 0:
                # batches = np.zeros((0, self.dims[0], self.dims[1], 6), dtype=np.float32)
                fn_im_paths = self.get_data_paths(img_dirs)
                np.random.shuffle(fn_im_paths)
                motion_1_share = params.get('augment_type_percent')[0] / sum(params.get('augment_type_percent')[:2])
                motion_2_share = params.get('augment_type_percent')[1] / sum(params.get('augment_type_percent')[:2])
                batches_augmented = self.augmentation_3D(fn_im_paths,
                                                      [motion_1_share, motion_2_share],
                                                      params.get('flow_amplitude'),
                                                      selected_frames,
                                                      selected_slices,
                                                      given_u=False,
                                                      max_num_to_take=augmented_data_num)
                batches_augmented = batches_augmented[..., :4]  # no undersampling
                batches_augmented = np.concatenate((batches_augmented, batches), axis=0)
            if not params.get('whole_kspace_training'):
                radius = int((params.get('crop_size') - 1) / 2)
                # if params.get('downsampling'):
                #     batches_augmented[..., 2] = downsampling_2D(batches_augmented[..., 2])  # TODO
                if params.get('padding'):
                    batches_augmented = np.pad(batches_augmented,
                                               ((0, 0), (radius, radius), (radius, radius), (0, 0)),
                                               constant_values=0)
                if params.get('random_crop'):
                    batches_augmented = self.crop2D(batches_augmented, crop_size=params.get('crop_size'), box_num=200)
                else:
                    x_dim, y_dim = np.shape(batches_augmented)[1:3]
                    pos = pos_generation_2D(intervall=[[0, x_dim - params.get('crop_size') + 1],
                                                       [0, y_dim - params.get('crop_size') + 1]], stride=4)
                    batches_augmented = self.crop2D_FixPts(batches_augmented,
                                                           crop_size=params.get('crop_size'),
                                                           box_num=np.shape(pos)[1], pos=pos)

                batches = np.concatenate((arr2kspace(batches_augmented[..., :2]), batches_augmented[..., 2:]), axis=-1)
                np.random.shuffle(batches)

                flow = batches[:, radius, radius, 4:6]

                # # take image domain as input, don't forget to change the queue channel number
                # batches = batches_augmented
                # np.random.shuffle(batches)
                # radius = int((params.get('crop_size') - 1) / 2)
                # flow = batches[:, radius, radius, 2:4]

                # np.save('/home/jpa19/PycharmProjects/MA/UnFlow/same_data_for_test', batches)
                # batches = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/same_data_for_test.npy')
                im1_queue = tf.train.slice_input_producer([batches[..., :2]], shuffle=False,
                                                          capacity=len(list(batches[..., 0])), num_epochs=None)
                im2_queue = tf.train.slice_input_producer([batches[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batches[..., 1])), num_epochs=None)
                flow_queue = tf.train.slice_input_producer([flow], shuffle=False,
                                                           capacity=len(list(flow)), num_epochs=None)

                return tf.train.batch([im1_queue, im2_queue, flow_queue],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads)
            else:  # whole k-space as input, fft(ux, uy) as label  TODO: a better solution for finding a proper label
                batches_augmented = arr2kspace(batches_augmented)  # 4 (im1, im2, ux, uy)=> 8 channels
                if params.get('random_crop'):
                    batches = self.crop2D(batches_augmented, crop_size=33, box_num=200)
                np.random.shuffle(batches)
                radius = int((params.get('crop_size') - 1) / 2)
                flow_k = batches[:, radius, radius, 4:8]

                im1_queue = tf.train.slice_input_producer([batches[..., :2]], shuffle=False,
                                                          capacity=len(list(batches[..., 0])), num_epochs=None)
                im2_queue = tf.train.slice_input_producer([batches[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batches[..., 1])), num_epochs=None)
                flow_queue = tf.train.slice_input_producer([flow_k], shuffle=False,
                                                           capacity=len(list(flow_k)), num_epochs=None)
                return tf.train.batch([im1_queue, im2_queue, flow_queue],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads)

    def input_patch_test_data(self, config):
        test_types = config['test_types'][0]
        img_dir = config['test_dir']
        img_dir_matlab_simulated = config['test_dir_matlab_simulated']
        selected_frames = config['selected_frames']
        selected_slices = config['selected_slices']
        amplitude = config['amplitude']
        cross_test = config['cross_test']
        given_u = config['given_u']

        fn_im_paths = self.get_data_paths(img_dir)
        orig_batch = self.augmentation_3D(fn_im_paths,
                                          [1-test_types, test_types],
                                          amplitude,
                                          selected_frames,
                                          selected_slices,
                                          given_u=given_u,
                                          cross_test=config['cross_test'])
        #orig_batch[..., :2] = orig_batch[..., 4:]  # no undersampling

        # fn_im_paths = self.get_data_paths(img_dir_matlab_simulated)
        # orig_batch = self.load_real_simulated_data(fn_im_paths, selected_slices, max_num_to_take=1)

        radius = int((config.get('crop_size') - 1) / 2)
        if config.get('padding'):
            orig_batch = np.pad(orig_batch,
                                ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

        x_dim, y_dim = np.shape(orig_batch)[1:3]
        pos = pos_generation_2D(intervall=[[0, x_dim - config['crop_size'] + 1],
                                           [0, y_dim - config['crop_size'] + 1]], stride=config['crop_stride'])

        batches_cp = self.crop2D_FixPts(orig_batch, crop_size=config['crop_size'], box_num=np.shape(pos)[1], pos=pos)
        batches_cp = np.concatenate((arr2kspace(batches_cp[..., :2]), batches_cp[..., 2:4]), axis=-1)
        flow = batches_cp[:, radius, radius, 4:6]

        im1_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., :2]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 0])), num_epochs=None)
        im2_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 1])), num_epochs=None)
        flow_patch_gt = tf.train.slice_input_producer([flow], shuffle=False,
                                                      capacity=len(flow), num_epochs=None)

        im1 = orig_batch[..., 0]
        im2 = orig_batch[..., 1]
        flow_orig = orig_batch[..., 2:4]
        im1_orig = orig_batch[..., 4]
        im2_orig = orig_batch[..., 5]
        test_batch = tf.train.batch([im1_patch_k_queue, im2_patch_k_queue, flow_patch_gt],
                                    batch_size=self.batch_size,
                                    num_threads=self.num_threads)

        return test_batch, im1, im2, flow_orig, im1_orig, im2_orig, np.transpose(pos)

    def input_test_data(self, config):

        test_types = config['test_types']
        img_dir = config['test_dir']
        img_dir_matlab_simulated = config['test_dir_matlab_simulated']
        selected_frames = config['selected_frames']
        selected_slices = config['selected_slices']
        amplitude = config['amplitude']
        crop = config['crop']
        test_in_kspace = config['test_in_kspace']
        cross_test = config['cross_test']
        given_u = config['given_u']

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
                                          selected_slices,
                                          given_u=given_u)

                # batch: batch_size * im_size[0] * im_size[1] * 8,
                # 8: [fft.real &  imag of im1, fft.real &  imag of im2, u1, u2, im1, im2]
                crop_size = 33
                x = np.arange(0, self.dims[0] - crop_size, 4)
                y = np.arange(0, self.dims[0] - crop_size, 4)
                vx, vy = np.meshgrid(x, y)
                vx = vx.reshape(vx.shape[1] * vx.shape[0])
                vy = vy.reshape(vy.shape[1] * vy.shape[0])
                pos = np.stack((vx, vy), axis=0)

                if crop:
                    batch = self.crop2D(batch, crop_size=64, box_num=1)
                    # batch = self.crop2D_FixPts(batch, crop_size=crop_size, box_num=np.shape(pos)[1], pos=pos)
                batch = np.concatenate((arr2kspace(batch[..., :2]), batch[..., 2:], batch[..., :2]), axis=-1)

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
                batch = self.augmentation_3D(fn_im_paths,
                                             [0, 1],
                                             amplitude,
                                             selected_frames,
                                             selected_slices,
                                             given_u=given_u)
                batch = batch[0, ...]  # only take the first sample
                batch = batch[np.newaxis, ...]
                batch[..., :2] = batch[..., 4:]  # If no undersampling
                batch = batch[..., :4].astype(dtype=np.float32)  # TODO!

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

    def test_2D_slice(self, config):

        batch = self.test_set_generation(config)
        batch = batch[np.newaxis, ...]

        radius = int((config['crop_size'] - 1) / 2)
        if config['padding']:
            batch = np.pad(batch, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
        x_dim, y_dim = np.shape(batch)[1:3]
        pos = pos_generation_2D(intervall=[[0, x_dim - config['crop_size'] + 1],
                                           [0, y_dim - config['crop_size'] + 1]], stride=config['crop_stride'])

        batches_cp = self.crop2D_FixPts(batch, crop_size=config['crop_size'], box_num=np.shape(pos)[1], pos=pos)
        batches_cp = np.concatenate((arr2kspace(batches_cp[..., :2]), batches_cp[..., 2:4]), axis=-1)
        flow = batches_cp[:, radius, radius, 4:6]

        im1_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., :2]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 0])), num_epochs=None)
        im2_patch_k_queue = tf.train.slice_input_producer([batches_cp[..., 2:4]], shuffle=False,
                                                          capacity=len(list(batches_cp[..., 1])), num_epochs=None)
        flow_patch_gt = tf.train.slice_input_producer([flow], shuffle=False,
                                                      capacity=len(flow), num_epochs=None)

        im1 = batch[..., 0]
        im2 = batch[..., 1]
        flow_orig = batch[..., 2:4]

        test_batch = tf.train.batch([im1_patch_k_queue, im2_patch_k_queue, flow_patch_gt],
                                    batch_size=self.batch_size,
                                    num_threads=self.num_threads)

        return test_batch, im1, im2, flow_orig, np.transpose(pos)

    def test_set_generation(self, config):
        path = config['path']
        frame = config['frame']
        slice = config['slice']
        u_type = config['u_type']
        use_given_u = config['use_given_u']
        US = config['US']
        US_acc = config['US_acc']
        use_given_US_mask = config['use_given_US_mask']

        paths = self.get_data_paths(path)

        if u_type == 2:
            if 'simu' in paths[0]:
                dset = load_mat_file(paths[0])
            else:
                dset = load_mat_file(paths[1])
            ref = np.array(dset['I1_real'], dtype=np.float32)
            mov = np.array(dset['I1_Real_hat'], dtype=np.float32)
            u0 = np.array(-dset['u_Real_est_1'], dtype=np.float32)
            u1 = np.array(-dset['u_Real_est_2'], dtype=np.float32)
            if US:
                if US_acc == 30:
                    mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc30.npy')
                elif US_acc == 8:
                    mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc8.npy')
                elif US_acc == 1:
                    mask = np.transpose(generate_mask(nSegments=25, acc=1, nRep=4), (2, 1, 0))
                # mask = np.transpose(generate_mask_2(subsampleType=1, acc=4, vd_type=1, nRep=4), (2, 1, 0))
                # acc = np.random.choice(np.arange(1, 32, 6))
                # acc = 30
                # mask = np.transpose(generate_mask(nSegments=25, acc=acc, nRep=4), (2, 1, 0))
                k_dset = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
                k_warped_dset = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
                ref = (np.fft.ifftn(k_dset)).real
                mov = (np.fft.ifftn(k_warped_dset)).real

            im1, im2, u0, u1 = ref[..., slice], mov[..., slice], u0[..., slice], u1[..., slice]
            im1, im2, u0, u1 = im1[np.newaxis, ...], im2[np.newaxis, ...], u0[np.newaxis, ...], u1[np.newaxis, ...]
            batch = np.concatenate((im1, im2, u0, u1), axis=0)
            batch = np.moveaxis(batch, 0, -1)
        elif u_type == 0 or u_type == 1:
            if 'simu' not in paths[0]:
                dset = load_mat_file(paths[0])
            else:
                dset = load_mat_file(paths[1])
            dset = np.array(dset['dImg'], dtype=np.float32)
            dset = np.transpose(dset, (3, 2, 1, 0))

            dset = (dset - np.amin(dset)) / (np.amax(dset) - np.amin(dset))
            dset = dset[..., frame]
            ref = np.rot90(dset, axes=(0, 1))
            img_size = np.shape(dset)
            if u_type == 0:
                u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/u_constant_amp10_3D.npy')
            elif u_type == 1:
                u = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/u_smooth_apt10_3D.npy')
            mov = np_warp_3D(ref, u)
            if US:
                if US_acc == 30:
                    mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc30.npy')
                elif US_acc == 8:
                    mask = np.load('/home/jpa19/PycharmProjects/MA/UnFlow/mask_acc8.npy')
                elif US_acc == 0:
                    mask = np.transpose(generate_mask(nSegments=25, acc=0, nRep=4), (2, 1, 0))
                # acc = np.random.choice(np.arange(1, 32, 6))
                # acc = 30
                # mask = np.transpose(generate_mask(nSegments=25, acc=acc, nRep=4), (2, 1, 0))
                k_ref = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
                k_mov = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
                ref = (np.fft.ifftn(k_ref)).real
                mov = (np.fft.ifftn(k_mov)).real
            im1, im2 = ref[..., np.newaxis], mov[..., np.newaxis]

            batch = np.concatenate((im1, im2, u[..., :2]), axis=-1)
            batch = np.transpose(batch, (2, 0, 1, 3))
            batch = batch[slice, ...]
        return batch


# for original Kitti data
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
