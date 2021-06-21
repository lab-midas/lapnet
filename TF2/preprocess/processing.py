import matplotlib.pyplot as plt
import numpy as np
import math
from core.image_warp import np_warp_3D
import scipy.io as sio
from core.undersample.sampling import generate_mask
from pyexcel_ods import get_data


def load_scaled_ref_img(ID, normalized, ImgPath='/mnt/data/rawdata/MoCo/LAPNet/resp/motion_data'):
    """ read reference image ans scale it if normalized is True"""
    if normalized:
        temp_data = np.load(ID)
        res_end = np.asarray(temp_data['dFixed'], dtype=np.float32)
    else:
        temp_data = sio.loadmat(f'{ImgPath}/{ID}_img.mat')
        res = temp_data['dImgC'][:, :, :, 0]
        max_amp = np.max(np.abs(res))
        res_real = res.real / max_amp
        res_im = res.imag / max_amp
        res_end = res_real + 1j * res_im
    return res_end


def plot_img(arr):
    plt.imshow(np.abs(arr))
    plt.show()


def load_scaled_u(ID, normalized=False, FlowPath='/mnt/data/rawdata/MoCo/LAPNet/resp/LAP', masked=False):
    """ read motion field and mask it if masked is True"""
    if normalized:
        temp_data = np.load(ID)
    else:
        temp_data = sio.loadmat(f'{FlowPath}/{ID}.mat')

    ux = np.asarray(temp_data['ux'], dtype=np.float32)
    uy = np.asarray(temp_data['uy'], dtype=np.float32)
    uz = np.asarray(temp_data['uz'], dtype=np.float32)
    u = np.stack((ux, uy, uz), axis=-1)

    if masked:
        mask = temp_data['lMask']
        mask_true = np.logical_not(mask).astype(np.uint8)
        for i in range(3):
            u[..., i] = np.multiply(u[..., i], mask_true)

    return ux, u


def flow_variation(ux, u_full, aug_type, amp=10):
    """ generate flows depending on the specified aug_type"""
    if aug_type == 'real_x_smooth':
        u_syn = _u_generation_3D(np.shape(ux), amp, motion_type=1)
        u = np.multiply(u_full, u_syn)
    elif aug_type == 'smooth':
        u = _u_generation_3D(np.shape(ux), amp, motion_type=1)
    elif aug_type == 'constant':
        u = _u_generation_3D(np.shape(ux), amp, motion_type=0)
    elif aug_type == 'real':
        u = u_full
    else:
        raise ImportError('wrong augmentation type is given')
    return u


def select_2D_Data(ref, mov, u, layer, direction):
    """ select 2D information for given 3D reference and moving images in the wanted direction
    :param ref: the reference 3D image
    :param mov: the corresponding moving 3D image
    :param u : the motion field
    :param layer : the slicing layer number
    :param direction : the slicing direction, 'coronal', 'sagittal' or 'axial'
    :return 4D array containing the 2D ref, mov and field
    """
    u1 = u[..., 0]
    u2 = u[..., 1]
    index = 2
    if direction != 'coronal':
        u2 = u[..., 2]
        index = 1
    if direction == 'axial':
        u1 = u[..., 1]
        index = None
    data_3D = np.stack((ref, mov, u1, u2), axis=-1)
    if index is not None:
        data_3D = np.moveaxis(data_3D, index, 0)
    Imgs = data_3D[layer, ...]
    return Imgs


def undersample(ref, mov, acc, mask_type, normalized=False):
    if mask_type == 'drUS':
        size_y = ref.shape[1]
        size_z = ref.shape[2]
        mask = np.transpose(generate_mask(acc=acc, size_y=size_y, nRep=4, numPar=size_z), (2, 1, 0))
        k_ref = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
        k_mov = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))

        ref = np.fft.ifftn(k_ref)
        mov = np.fft.ifftn(k_mov)
        if normalized:
            ref = ref.real
            mov = mov.real
    return ref, mov


def load_data_3D(dataID, img_path, flow_path, aug_type, us_rate, mask_type, normalized, masking, amp=10):
    """load the reference image and generate the moving image and the corresponding motion field
    :param dataID: the reference image ID
    :param img_path: reference image file path
    :param flow_path : simulated LAP flow path
    :param aug_type : flow augementation type, 'smooth', 'constant', 'real' or 'real_x_smooth'
    :param us_rate : the undersampling rate
    :param normalized : specify if the given reference image is normalized or not
    :param amp : smooth flow augmentation amplitude
    :return list of the undersampled reference and moving images
    """
    # read data
    ref_3D = load_scaled_ref_img(dataID, normalized, img_path)
    ux, u_full = load_scaled_u(dataID, normalized, flow_path, masked=masking)
    if ref_3D.shape != ux.shape:
        u_full = np.swapaxes(u_full, 0, 1)
        ux = np.swapaxes(ux, 0, 1)
    # augment motion field
    u_3D = flow_variation(ux, u_full, aug_type, amp)
    # warping
    if normalized:
        mov_3D = np_warp_3D(ref_3D, u_3D)
    else:
        mov_3D_real = np_warp_3D(ref_3D.real, u_3D)
        mov_3D_img = np_warp_3D(ref_3D.imag, u_3D)
        mov_3D = mov_3D_real + 1j * mov_3D_img
    # downsample data
    if us_rate !=1:
        ref_3D, mov_3D = undersample(ref_3D, mov_3D, us_rate, mask_type, normalized)

    return ref_3D, mov_3D, u_3D, us_rate


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


def get_slice_info_from_ods_file(info_file):
    ods = get_data(info_file)
    slice_info = {value[0]: list(range(*[int(j) for j in value[1].split(',')])) for value in ods["Sheet1"] if
                  len(value) is not 0}
    return slice_info


def get_maxmin_info_from_ods_file(info_file):
    ods = get_data(info_file)
    slice_info = {value[0]: list(value[1].split(',')) for value in ods["Sheet1"] if len(value) is not 0}
    return slice_info
