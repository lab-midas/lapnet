import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


def fft_along_dim(x):
    """give the FFT along rows and columns of a 2D array
    :param x: input 2d array
    :return: 2d array
    """
    res = np.apply_along_axis(np.fft.fft, 0, x, norm="ortho")
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


def np_fftconvolve(A, B):
    return np.fft.ifft2(np.multiply(np.fft.fft2(A), np.fft.fft2(B, s=A.shape)))


def ifftnshift(x):
    """give shifted IFFT of x
    :param x: 2D or 3D array
    :return: 2D or 3D array
    """
    res = x
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
    x = np.linspace(0, rows - 1, num=ix.shape[0] + 1)
    y = np.linspace(0, cols - 1, num=iy.shape[0] + 1)
    grid_xq, grid_yq = np.meshgrid(x, y)
    regridded_k = interpn((grid_x, grid_y), k, (grid_yq, grid_xq), method='linear')
    return regridded_k


def plotfullsampledkspace(k):
    """show image corresponding to a full sampled k space
    :param k: 2D array
    :return: plot image
    """
    img = ifftnshift(k)
    plt.imshow(np.abs(img))
    plt.show()


def plottaperedkspace(k):
    """show image corresponding to a tapered k space
    :param k: 2D array
    :return: plot image
    """
    image = ifft_along_dim(k)
    plt.imshow(np.abs(image))
    plt.show()


def pad_img(img, ix, iy):
    """transform k to time domain pad it so that the cropped part defined by ix and iy is in
    the center of a square array and return it to k space for a central crop
    :param img: 2D array to be padded
    :param ix: 1D array of the cropping along x interval
    :param iy: 1D array of the cropping along y interval
    :return: 2d padded k space + new cropping intervals
    """
    # img = ifftnshift(k)
    rows, cols = img.shape[-2], img.shape[-1]
    centercropx = ix[0] + round((ix.shape[0] + 1) / 2)
    centercropy = iy[0] + round((iy.shape[0] + 1) / 2)
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

    ix = ix + padxbefore
    iy = iy + padybefore
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

    if im.shape[0] > im.shape[1]:
        extra = int(round(im.shape[0] - im.shape[1]) / 2)
        im = np.pad(im, ((0, 0), (extra, extra)), 'constant')
        iy = iy + extra

    if im.shape[0] < im.shape[1]:
        extra = int(round(im.shape[1] - im.shape[0]) / 2)
        im = np.pad(im, ((extra, extra), (0, 0)), 'constant')
        ix = ix + extra

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

    return k_win


def reshuffle(k):
    """reshuffle k
    :param k: 2d array to be reshuffled
    :return: 2d reshuffled array
    """
    k[::2, :] = -k[::2, :]
    k[:, ::2] = -k[:, ::2]
    return k


def taper2D(ref_img, mov_img, x_pos, y_pos, crop_size=33, u=None):
    k_ref = rectangulartapering2d(ref_img, x_pos, y_pos, crop_size)
    k_mov = rectangulartapering2d(mov_img, x_pos, y_pos, crop_size)
    k_ref = np.stack((np.real(k_ref), np.imag(k_ref), np.real(k_mov), np.imag(k_mov)), axis=-1)
    if u is not None:
        flow = flowCrop(u, x_pos, y_pos, crop_size)
        return k_ref, flow
    else:
        return k_ref


def rectangulartapering2d(ki, x, y, p):
    """perform tapering in fourier domain
    :param ki: initial 2d image to be tapered
    :param x: position of cropping along the x axis
    :param y: position of cropping along the y axis
    :param p: length of the cropped piece (pxp)
    :return: tapered 2d k_space
    """
    # ix: 1D array of the cropping along x interval
    idx = np.arange(x, x + p - 1)
    # iy: 1D array of the cropping along y interval
    idy = np.arange(y, y + p - 1)
    k, idx, idy = pad_img(ki, idx, idy)
    # plotfullsampledkspace(k)
    k_win = RectWindow(k, idx, idy)
    k_cut = np_fftconvolve(k, k_win)
    # plotfullsampledkspace(k_cut)
    regridded_k_space = regrid(k_cut, idx, idy)
    # plottaperedkspace(regridded_k_space)
    tapered_k_space = reshuffle(regridded_k_space)
    # plottaperedkspace(tapered_k_space)
    # tapered_k_space = np.stack((np.real(tapered_k_space), np.imag(tapered_k_space)), axis=-1)
    return tapered_k_space


def flowCrop(arr, x, y, p):
    radius = int((p - 1) / 2)
    xp = x + radius
    yp = y + radius
    return arr[xp, yp, :].real

