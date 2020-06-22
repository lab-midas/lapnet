import tensorflow as tf
import os
import math
import scipy.io as sio
import h5py
from ..ops import downsample as downsample_ops
import numpy as np
from pathlib import Path
import matplotlib
# matplotlib.use('pdf')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))


def downsample(tensor, num):
    _,height, width,_ = tensor.shape.as_list()
    if height%2==0 and width%2==0:
        return downsample_ops(tensor, num)
    else:
        return tf.image.resize_area(tensor, tf.constant([int(height/num), int(width/num)]))


def mat2npz(mat_dirs, output_dirs, save_type=np.float32):
    if not os.path.exists(output_dirs):
        os.mkdir(output_dirs)
    mat_files = os.listdir(mat_dirs)
    for mat_file in mat_files:
        f = load_mat_file(os.path.join(mat_dirs, mat_file))
        keys2save = [i for i in f.keys() if isinstance(f[i], np.ndarray)]
        values2save = [np.asarray(f[i], dtype=save_type) if f[i].dtype == np.float64 else f[i] for i in keys2save]
        save_path = os.path.join(output_dirs, mat_file.split('.')[0])
        np.savez(save_path, **{name: value for name, value in zip(keys2save, values2save)})

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


def save_img(result, file_path, format='png'):
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.axis('off')
    if len(result.shape) == 2:
        plt.imshow(result, cmap="gray")
    else:
        plt.imshow(result)
    fig.savefig(file_path+'.'+format)
    plt.close()


def save_imgs_with_arrow(im1, im2, u):
    x = np.arange(0, 256, 8)
    y = np.arange(0, 256, 8)
    x, y = np.meshgrid(x, y)
    # matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')
    # to make it consistent with np_warp, ux should be negative
    ax[0].quiver(x, y, u[0:256:8, :, :][:, 0:256:8, :][:, :, 0],
                 -u[0:256:8, :, :][:, 0:256:8, :][:, :, 1], color='y')
    fig.savefig('/home/jpa19/PycharmProjects/MA/UnFlow/data/resp/' + 'a' + '.' + 'png')
    plt.close(fig)


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