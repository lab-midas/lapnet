import math
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import merlintf
import random

from core import np_warp_3D, np_warp_2D, generate_mask, subsample_radial, post_crop, pos_generation_2D, taper2D, rectangulartapering2d


############
# load data
############

def load_4d_img(f_name, path):
    img = sio.loadmat(f'{path}/{f_name}_img.mat')['dImgC']
    max_amp = np.max(np.abs(img))
    img = (img.real / max_amp) + 1j * (img.imag / max_amp)
    return img


def select_2d_data(ref, mov, u, layer, direction):
    """ select 2D information for given 3D reference and moving images in the wanted direction
    :param ref: the reference 3D image
    :param mov: the corresponding moving 3D image
    :param u : the motion field
    :param layer : the slicing layer number
    :param direction : the slicing direction, 'coronal', 'sagittal' or 'axial'
    :return list containing the selected ref, mov and field
    """
    assert (direction in ['coronal', 'sagittal', 'axial'], "given data selection direction is not supported")
    if direction == 'coronal':
        u1 = u[..., 0]
        u2 = u[..., 1]
        index = 2
    if direction == 'sagittal':
        u1 = u[..., 0]
        u2 = u[..., 2]
        index = 1
    if direction == 'axial':
        u1 = u[..., 1]
        u2 = u[..., 2]
        index = 0
    u = np.stack((u1, u2), axis=-1)
    ref = np.moveaxis(ref, index, 0)[layer, ...]
    mov = np.moveaxis(mov, index, 0)[layer, ...]
    u = np.moveaxis(u, index, 0)[layer, ...]
    return ref, mov, u


############
# flow augmentation
############

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

    return np.asarray(u, dtype=np.float32)


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
        # u_C = 2 * np.random.rand(2)
        u_C = -1 + 2 * np.random.rand(2)  # interval [-1, 1]
        amplitude = amplitude / np.linalg.norm(u_C, 2)
        u = amplitude * np.ones((M, N, 2))
        u[..., 0] = u_C[0] * u[..., 0]
        u[..., 1] = u_C[1] * u[..., 1]
        pass
    elif motion_type == 1:
        u = -1 + 2 * np.random.rand(M, N, 2)
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


def flow_variation_2d(uxy, s, aug_type):
    if aug_type == 'real_x_smooth':
        u_syn = _u_generation_2D(np.shape(uxy)[:2], amplitude=5, motion_type=1)
        u = np.multiply(uxy[:, :, s], u_syn)
    elif aug_type == 'smooth':
        u = _u_generation_2D(np.shape(uxy)[:2], amplitude=5, motion_type=1)
    elif aug_type == 'constant':
        u = _u_generation_2D(np.shape(uxy)[:2], amplitude=5, motion_type=0)
    elif aug_type == 'real':
        u = uxy[:, :, s]
    else:
        raise ImportError('wrong augmentation type is given')
    return u


def flow_variation_3d(u_full, aug_type, amp):
    """ generate flows depending on the specified aug_type"""
    shape = np.shape(u_full)[:-1]
    if aug_type == 'real_x_smooth':
        u_syn = _u_generation_3D(shape, amp, motion_type=1)
        u = np.multiply(u_full, u_syn)
    elif aug_type == 'smooth':
        u = _u_generation_3D(shape, amp, motion_type=1)
    elif aug_type == 'constant':
        u = _u_generation_3D(shape, amp, motion_type=0)
    elif aug_type == 'real':
        u = u_full
    else:
        raise ImportError('wrong augmentation type is given')
    return u


############
# Subsampling
############


def subsample_img_drus(img, acc, n_frame=0):
    """ subsample 3D image with the CASPR undersampling given an defined acceleration
        :param img: fully sampled 3D image
        :param acc: undersampling factor
        :return subsampled 3d image
    """
    mask = np.transpose(generate_mask(acc=acc, size_y=img.shape[1], nRep=4, numPar=img.shape[2]), (2, 1, 0))
    k = np.multiply(np.fft.fftn(img), np.fft.ifftshift(mask[n_frame, ...]))
    sub_img = np.fft.ifftn(k)
    return sub_img


def subsample_respiratory_data(ref, mov, acc, mask_type):
    assert (mask_type in ['drUs', 'radUs'], "given subsampling mask is not supported")
    real_data = np.isreal(ref)
    if acc != 1:
        if mask_type == 'drUs':
            ref = subsample_img_drus(ref, acc, n_frame=0)
            mov = subsample_img_drus(mov, acc, n_frame=3)
            if real_data == 'real':
                ref = ref.real
                mov = mov.real

        elif mask_type == 'radUs':
            im_pair = np.stack((ref, mov), axis=-1)
            im_pair_US = subsample_radial(im_pair, acc=acc)
            im_pair = post_crop(im_pair_US, np.shape(im_pair))
            ref, mov = im_pair[..., 0], im_pair[..., 1]
            if real_data == 'real':
                ref = np.abs(ref)
                mov = np.abs(mov)
    return ref, mov


def subsample_cardiac_data(img_pair_batch, smaps, acc, mask_root_dir):
    sd = random.randint(1, 20)
    p = img_pair_batch.shape[1]
    mask_last_name = 'mask_VISTA_%dx%d_acc%d_%d.txt' % (p, 25, acc, sd)
    mask = np.loadtxt(f'{mask_root_dir}/{mask_last_name}', dtype=int, delimiter=",")
    mask_ref, mask_mov = mask[:, 0], mask[:, -1]
    mask = np.expand_dims(np.stack((mask_ref, mask_mov), axis=0), (0, 2, 4))

    # create input k-space
    imgccoil = img_pair_batch * np.conj(smaps)
    coilkspace = fft2c(imgccoil)

    # apply mask
    masked_kspace = mask * coilkspace
    masked_coilimg = ifft2c(masked_kspace)
    masked_img = tf.reduce_sum(masked_coilimg * smaps, -1, keepdims=True)
    masked_img = np.squeeze(masked_img)
    ref_img, mov_img = masked_img[0], masked_img[1]

    return ref_img, mov_img


############
# Fourier transformations for VISTA undersampling
############

def fft2c(image):
    image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
    dtype = tf.math.real(image).dtype
    axes = [tf.rank(image) - 2, tf.rank(image) - 1]  # two most inner dimensions
    scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
    kspace = merlintf.complex_scale(
        tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(image, axes=axes)), axes=axes), 1 / scale)
    return tf.transpose(kspace, perm=[0, 1, 3, 4, 2])


def ifft2c(kspace):
    kspace = tf.transpose(kspace, perm=[0, 1, 4, 2, 3])
    dtype = tf.math.real(kspace).dtype
    axes = [tf.rank(kspace) - 2, tf.rank(kspace) - 1]
    scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
    image = merlintf.complex_scale(
        tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(kspace, axes=axes)), axes=axes), scale)
    return tf.transpose(image, perm=[0, 1, 3, 4, 2])


############
# sliding tapering
############
def sliding_tapering(k_ref, k_mov, k_u, npz_saving_path, tapering_size, crop_stride):
    radius = int((tapering_size - 1) / 2)
    ref = np.pad(k_ref, (radius, radius), constant_values=0)
    mov = np.pad(k_mov, (radius, radius), constant_values=0)

    u = np.pad(k_u, ((radius, radius), (radius, radius), (0, 0)), constant_values=0)
    x_dim, y_dim = np.shape(ref)
    pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                       [0, y_dim - tapering_size + 1]], stride=crop_stride)
    print(pos.shape)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4), dtype=np.float32)
    flow_output = np.zeros((pos.shape[1], 2), dtype=np.float32)

    # Generate tapered data
    for i in range(pos.shape[1]):
        pos_tmp = pos[:, i]
        # print(pos_tmp)
        k_output[i], flow_output[i, :] = taper2D(ref, mov,
                                                 pos_tmp[0],
                                                 pos_tmp[1],
                                                 tapering_size,
                                                 u=u)
    np.savez(npz_saving_path,
             k_ref=k_ref,
             k_mov=k_mov,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)
    print(f'{os.path.basename(npz_saving_path)}: is saved')


def sliding_tapering_one_frame(img_2d, save_path, tapering_size, crop_stride):
    x_dim, y_dim = np.shape(img_2d)
    pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                       [0, y_dim - tapering_size + 1]], stride=crop_stride)
    print(pos.shape)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size), dtype=np.float32)

    for i in range(pos.shape[1]):
        pos_tmp = pos[:, i]
        k_output[i] = rectangulartapering2d(img_2d, pos_tmp[0], pos_tmp[1], tapering_size)

    np.save(save_path, k_output)
    print(save_path, 'is saved')


def save_patches_along_depth(ref_img, mov_img, u, box_num, saving_dir, taper_size, ref_full, mov_full):
    x_dim, y_dim = ref_img.shape

    x_pos = np.random.randint(0, x_dim - taper_size + 1, box_num)
    y_pos = np.random.randint(0, y_dim - taper_size + 1, box_num)

    for num in range(box_num):
        train_kspace, train_flow = taper2D(ref_img, mov_img, x_pos[num], y_pos[num], u=u, crop_size=taper_size)
        fully_sampled_k = taper2D(ref_full, mov_full, x_pos[num], y_pos[num], crop_size=taper_size)
        np.savez(f'{saving_dir}_{num + 1}.npz',
                 k_space=train_kspace,
                 k_full=fully_sampled_k,
                 flow=train_flow)


def taper_selected_slice(ref_3D, mov_3D, u_3D, pos, direction='coronal', offset=None, crop_size=33):
    x_pos, y_pos, z_pos = pos
    ref, mov, u = select_2d_data(ref_3D, mov_3D, u_3D, z_pos, direction)

    if offset:
        ref = np.pad(ref, (offset, offset), constant_values=0)
        mov = np.pad(mov, (offset, offset), constant_values=0)

    ref_tapered, mov_tapered = taper2D(ref, mov, x_pos, y_pos, crop_size=crop_size)

    return ref_tapered, mov_tapered


############
# datasets
############

class respiratoryDataset():
    def __init__(self, args):
        self.args = args

    def load_respiratory_img(self, f_name):
        """ read 4d image """
        if self.args.data_type is 'real':
            img_dict = np.load(f'{self.args.img_path}/{f_name}.npz')
            img = np.asarray(img_dict['dFixed'], dtype=np.float32)
        elif self.args.data_type is 'complex':
            img = sio.loadmat(f'{self.args.img_path}/{f_name}_img.mat')['dImgC']
        else:
            raise Exception('given image type is unknown')
        return img

    def load_lap_flow(self, f_name):
        """ read motion field and mask it if masked is True"""
        if self.args.data_type is 'real':
            u_dict = np.load(f'{self.args.flow_path}/{f_name}.npz')
        elif self.args.data_type is 'complex':
            u_dict = sio.loadmat(f'{self.args.flow_path}/{f_name}.mat')
        else:
            raise Exception('given image type is unknown')

        ux = np.asarray(u_dict['ux'], dtype=np.float32)
        uy = np.asarray(u_dict['uy'], dtype=np.float32)
        uz = np.asarray(u_dict['uz'], dtype=np.float32)
        u = np.stack((ux, uy, uz), axis=-1)

        if self.args.masked_flow:
            mask = u_dict['lMask']
            mask_true = np.logical_not(mask).astype(np.uint8)
            for i in range(3):
                u[..., i] = np.multiply(u[..., i], mask_true)

        return u

    def read_respiratory_data(self, f_name, aug_type, acc):
        """load the reference image and generate the moving image and the corresponding motion field
        :param f_name: the reference image ID
        :param aug_type : flow augementation type, 'smooth', 'constant', 'real' or 'real_x_smooth'
        :param acc : the undersampling rate
        :return list of the 3D fully sampled + undersampled reference and moving images
        """
        # read data
        ref_full = self.load_respiratory_img(f_name)[..., 0]
        u_lap = self.load_lap_flow(f_name)

        if self.args.simulated:
            flow = flow_variation_3d(u_lap, aug_type, self.args.amp)
            # warping
            if self.args.data_type == 'real':
                mov_full = np_warp_3D(ref_full, flow)
            elif self.args.data_type == 'complex':
                mov_full = np_warp_3D(ref_full.real, flow) + 1j * np_warp_3D(ref_full.imag, flow)
        else:
            mov_full = self.load_respiratory_img(f_name)[..., 3]
            flow = u_lap

        # downsample data
        sub_ref, sub_mov = subsample_respiratory_data(ref_full, mov_full, acc, self.args.mask_type)
        flow = np.swapaxes(flow, 0, 1) if ref_full.shape != flow.shape[:-1] else flow

        return ref_full, mov_full, sub_ref, sub_mov, flow, acc


class cineDataset():
    def __init__(self, args):
        self.args = args

    def load_img(self, f_name):
        norm_imgc = np.load(f'{self.args.img_path}/norm_imgc_{f_name}.txt.npy')  # coil-combined
        batch_imgc = np.transpose(norm_imgc, [2, 3, 0, 1])
        return batch_imgc

    def load_map(self, f_name):
        averaged_smap = np.load(f'{self.args.img_path}/averaged_smap_cc_15_{f_name}.txt.npy')
        batch_smaps = np.expand_dims(np.transpose(averaged_smap, [2, 0, 1, 3]), axis=1)
        return batch_smaps

    def load_lap_flow(self, f_name):
        data = sio.loadmat(f'{self.args.flow_path}/{f_name}.mat')
        ux = np.asarray(data['ux'], dtype=np.float32)
        uy = np.asarray(data['uy'], dtype=np.float32)
        uxy = np.stack((ux, uy), axis=-1)
        return uxy

    def read_cardiac_data(self, f_name, aug_type, acc, slice_num):
        batch_imgc = self.load_img(f_name)
        batch_smaps = self.load_map(f_name)

        ref = batch_imgc[slice_num, 0, ...]
        smaps = batch_smaps[slice_num, None, ...]

        uxy = self.load_lap_flow(f_name)
        u = flow_variation_2d(uxy, slice_num, aug_type)
        mov = np_warp_2D(ref, u)
        img_pair_batch = np.expand_dims(np.stack((ref, mov), axis=0), axis=[0, -1])

        sub_ref, sub_mov = subsample_cardiac_data(img_pair_batch, smaps, acc, self.args.mask_root_dir)

        return ref, mov, sub_ref, sub_mov, u  # 2D
