import os
import scipy.io as sio
import h5py
import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt


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


def save_img(result, file_path, format='png'):
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.axis('off')
    if len(result.shape) == 2:
        plt.imshow(result, cmap="gray")
    else:
        plt.imshow(result)
    fig.savefig(file_path + '.' + format)
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


def save_ods_file(data, filename):
    from pyexcel_ods import save_data
    sheetx = {'Sheet1': data}
    save_data(f'{filename}.ods', sheetx)
