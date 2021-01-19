import tensorflow as tf
import os
import math
import scipy.io as sio
import h5py
# from ..ops import downsample as downsample_ops
import numpy as np
from pathlib import Path
import matplotlib
# matplotlib.use('pdf')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import operator
from scipy.interpolate import interpn
from scipy.signal import convolve2d


def summarized_placeholder(name, prefix=None, key=tf.compat.v1.GraphKeys.SUMMARIES):
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

"""
def downsample(tensor, num):
    _,height, width,_ = tensor.shape.as_list()
    if height%2==0 and width%2==0:
        return downsample_ops(tensor, num)
    else:
        return tf.image.resize_area(tensor, tf.constant([int(height/num), int(width/num)]))
"""

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


# def cal_loss_mean(loss_dir):
#     if os.path.exists(os.path.join(loss_dir, 'mean_loss.txt')):
#         raise ImportError('mean value already calculated')
#     files = os.listdir(loss_dir)
#     files.sort()
#     mean = dict()
#     for file in files:
#         with open(os.path.join(loss_dir, file), 'r') as f:
#             data = [float(i) for i in f.readlines()]
#             mean[file.split('.')[0]] = np.mean(data)
#     with open(os.path.join(loss_dir, 'mean_loss.txt'), "a") as f:
#         for name in mean:
#             f.write('{}:{}\n'.format(name, round(mean[name], 5)))

def central_crop(img, bounding):
    """
    central crop for 2D/3D arrays
    # alternative code:
    cutting_part = int((crop_size - 1)/2)
    flow_gt_cut = flow_gt[cutting_part:(np.shape(flow_gt)[0] - cutting_part - 1),
    cutting_part:(np.shape(flow_gt)[0] - cutting_part - 1), :]

    :param img:
    :param bounding:
    :return:
    """
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def cal_loss_mean(loss_dir):
    if os.path.exists(os.path.join(loss_dir, 'mean_loss_EPE.txt')):
        raise ImportError('mean value of EPE already calculated')
    if os.path.exists(os.path.join(loss_dir, 'mean_loss_EAE.txt')):
        raise ImportError('mean value of EAE already calculated')
    if os.path.exists(os.path.join(loss_dir, 'mean_loss.txt')):
        raise ImportError('mean value already calculated')
    files = os.listdir(loss_dir)
    files.sort()
    mean = dict()
    for file in files:
        with open(os.path.join(loss_dir, file), 'r') as f:
            data = [float(i) for i in f.readlines()]
            mean[file.split('.')[0]] = np.mean(data)
    f1 = open(os.path.join(loss_dir, 'mean_loss_EPE.txt'), "a")
    f2 = open(os.path.join(loss_dir, 'mean_loss_EAE.txt'), "a")
    for name in mean:
        if 'EPE' in name:
            f1.write('{}:{}\n'.format('_'.join(name.split('_')[:2]), round(mean[name], 5)))
        else:
            f2.write('{}:{}\n'.format('_'.join(name.split('_')[:2]), round(mean[name], 5)))
    f1.close()
    f2.close()

def fft_along_dim(x):
    """give the FFT along rows and columns of a 2D array
    :param x: input 2d array
    :return: 2d array
    """
    res = x
    res = np.apply_along_axis(np.fft.fft, 0, res, norm="ortho")
    res = np.apply_along_axis(np.fft.fft, 1, res, norm="ortho")
    return res


def ifft_along_dim(x):
    """give the IFFT along rows and columns of x
    :param x: input 2d array
    :return: 2d array
    """
    res = x
    res = np.apply_along_axis(np.fft.ifft, 0, res, norm="ortho")
    res = np.apply_along_axis(np.fft.ifft, 1, res, norm="ortho")
    return res


def fftnshift(x):
    """give shifted FFT of x
    :param x: 2D or 3D array
    :return: 2D or 3D array
    """
    res = np.zeros(x.shape, dtype=x.dtype)
    if x.ndim == 2:
        res = np.fft.fftshift(fft_along_dim(np.fft.ifftshift(x)))
    if x.ndim == 3:
        for i in np.arange(x.shape[2]):
            res[:, :, i] = np.fft.fftshift(fft_along_dim(np.fft.ifftshift(x[:, :, i])))
    if x.ndim == 4:
        res = np.zeros((np.shape(x)[0], np.shape(x)[1], np.shape(x)[2], 2), dtype=np.complex64)
        for i in np.arange(res.shape[-1]):
            for k in np.arange(res.shape[0]):
                res[k, :, :, i] = np.fft.fftshift(fft_along_dim(np.fft.ifftshift(x[k, :, :, i])))
    return res


def ifftnshift(x):
    """give shifted IFFT of x
    :param x: 2D or 3D array
    :return: 2D or 3D array
    """
    if x.ndim == 2:
        res = np.fft.fftshift(ifft_along_dim(np.fft.ifftshift(x)))
    if x.ndim == 3:
        res = np.zeros(x.shape, dtype=np.complex64)
        for i in np.arange(x.shape[2]):
            res[:, :, i] = np.fft.fftshift(ifft_along_dim(np.fft.ifftshift(x[:, :, i])))
    return res


def regrid(k, ix, iy):
    """regrid k depending on ix and iy
    :param k: 2D array to be regridded
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :return: 2D array
    """
    rows, cols = k.shape
    grid_x, grid_y = np.mgrid[0:rows], np.mgrid[0:cols]
    x = np.linspace(0, rows-1, num=ix.shape[0]+1)
    y = np.linspace(0, cols-1, num=iy.shape[0]+1)
    grid_xq, grid_yq = np.meshgrid(x, y)
    regridded_k = interpn((grid_x, grid_y), k, (grid_yq, grid_xq), method='linear')
    return regridded_k


def plotfullsampledkspace(k):
    """show image corresponding to a full sampled k space
    :param k: 2D array
    :return: plot image
    """
    image = ifftnshift(k)
    plt.imshow(np.abs(image))
    plt.show()


def plottaperedkspace(k):
    """show image corresponding to a tapered k space
    :param k: 2D array
    :return: plot image
    """
    image = ifft_along_dim(k)
    plt.imshow(np.abs(image))
    plt.show()

def paddimg(k, ix, iy):
    """transform k to time domain pad it so that the cropped part defined by ix and iy is in
    the center of a square array and return it to k space for a central crop
    :param k: 2D array to be padded
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :return: 2d padded k space + new cropping intervals
    """
    img = ifftnshift(k)
    rows, cols = img.shape
    centercropx = ix[0]+round((ix.shape[0]+1) / 2)
    centercropy = iy[0]+round((iy.shape[0]+1) / 2)
    padx = rows - 2 * centercropx
    pady = cols - 2 * centercropy
    if padx > 0:
        padxbefore = padx
        padxafter = 0
    else:
        padxbefore = 0
        padxafter = -padx
    if pady > 0:
        padybefore = pady
        padyafter = 0
    else:
        padybefore = 0
        padyafter = -pady
    im = np.pad(img, ((padxbefore, padxafter), (padybefore, padyafter)), 'constant')
    ix = ix+padxbefore
    iy = iy+padybefore

    paddingdifference = int(round((abs(padx) - abs(pady)) / 2))
    if paddingdifference > 0:
        padysquare = paddingdifference
        padxsquare = 0
        iy = iy + paddingdifference
    else:
        padysquare = 0
        padxsquare = -paddingdifference
        ix = ix - paddingdifference

    im = np.pad(im, ((padxsquare, padxsquare), (padysquare, padysquare)), 'constant')
    Kpadded = fftnshift(im)
    return Kpadded, ix, iy


def RectWindow(k, ix, iy):
    """
    :param k:  2D array in fourier domain to be windowed
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :return: 2D array
    """
    rows, cols = k.shape
    win = np.zeros((rows, cols), dtype=np.complex64)
    idx, idy = np.meshgrid(ix, iy)
    win[idx, idy] = 1
    k_win = fftnshift(win)
    kcut = convolve2d(k, k_win, mode='same')
    return kcut


def reshuffle(k):
    """reshuffle k
    :param k: 2d array to be reshuffled
    :return: 2d reshuffled array
    """
    k[::2, :] = -k[::2, :]
    k[:, ::2] = -k[:, ::2]
    return k


def rectangulartapering2d(ki, x, y, p):
    """perform tapering in fourier domain
    :param ki: initial 2d kspace to be tapered
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :return: tapered 2d k_space
    """
    # arr_kspace = np.zeros((np.shape(ki)[0], p, p, 2), dtype=np.float32)
    import medutils
    arr_kspace = np.zeros((p, p, 2), dtype=np.float32)
    ix = np.arange(x, x+p-1)
    iy = np.arange(y, y+p-1)
    k, idx, idy = paddimg(ki, ix, iy)
    k_cut = RectWindow(k, idx, idy)
    regridded_k_space = regrid(k_cut, idx, idy)
    tapered_k_space = reshuffle(regridded_k_space)
    arr_kspace[:, :, 0] = np.real(tapered_k_space)
    arr_kspace[:, :, 1] = np.imag(tapered_k_space)
    return arr_kspace


def rectangulartapering3d(k, ix, iy, iz):
    """perform tapering in fourier domain on 3d array
    :param k: initial 3d kspace to be tapered
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :param iz: 1D array of the cropping along z interval
    :return: tapered 3d k_space
    """
    x = ix.shape[0] + 1
    y = iy.shape[0] + 1
    z = iz.shape[0] + 1
    res = np.zeros((x, y, z), dtype=np.complex128)
    for i in np.arange(res.shape[2]):
        res[:, :, i] = rectangulartapering2d(k[:, :, i+iz[0]], ix, iy)
    return res


def flowCrop(arr, x, y, p):
    radius = int((p - 1) / 2)
    xp = x + radius
    yp = y + radius
    return arr[xp, yp, :]

def normalize_complex_arr(arr):
    a_oo = arr - arr.real.min() - 1j*arr.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()
