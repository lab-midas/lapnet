import numpy as np
from skimage.util import view_as_windows
from e2eflow.core.tapering import flowCrop


def crop2D(ref, mov, u, pos, crop_size):
    """crop a given array in time domain"""
    x_pos = pos[0]
    y_pos = pos[1]
    # cropping
    window_size = (crop_size, crop_size)
    ref_tmp = view_as_windows(ref, window_size)[x_pos, y_pos]
    mov_tmp = view_as_windows(mov, window_size)[x_pos, y_pos]
    ref_mov = np.stack((ref_tmp, mov_tmp), axis=-1)
    flow_out = flowCrop(u, x_pos, y_pos, crop_size)
    return ref_mov, flow_out


def crop2D_FixPts(arr, crop_size, box_num, pos):
    """crop a given array in time domain in fixed given positions"""
    arr_cropped_augmented = np.zeros((np.shape(arr)[0] * box_num, crop_size, crop_size, np.shape(arr)[-1]),
                                     dtype=np.float32)
    for batch in range(np.shape(arr)[0]):
        for i in range(box_num):
            arr_cropped_augmented[batch * box_num + i, ...] = arr[batch,
                                                              pos[0][i]:pos[0][i] + crop_size,
                                                              pos[1][i]:pos[1][i] + crop_size,
                                                              :]

    return arr_cropped_augmented


def arr2kspace(arr, normalize=False):
    """
    convert a 4D array (batch_size, x_dim, y_dim, channels) to kspace along the last axis, FFT on x and y dimension
    :param arr:
    :param normalize:
    :return: (batch_size, x_dim, y_dim, 2 * channel)
    """
    if arr.dtype == np.float64:
        arr = arr.astype(dtype=np.float32)
    arr_kspace = np.zeros((np.shape(arr)[0], np.shape(arr)[1], np.shape(arr)[2], 2 * np.shape(arr)[3]),
                          dtype=np.float32)
    for i in range(np.shape(arr)[-1]):
        kspace = to_freq_space(arr[..., i], normalize=normalize)
        arr_kspace[..., 2 * i:2 * i + 2] = kspace
    return arr_kspace


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
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
