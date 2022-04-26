import numpy as np
from skimage.transform import warp


def np_warp_3D(img, flow):

    img = img.astype('float32')
    flow = flow.astype('float32')
    height, width, thick = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    posx, posy, posz = np.mgrid[:height, :width, :thick]
    vx = flow[:, :, :, 1]  # to make it consistent as in matlab, ux in python is uy in matlab
    vy = flow[:, :, :, 0]
    vz = flow[:, :, :, 2]

    coord_x = posx + vx
    coord_y = posy + vy
    coord_z = posz + vz
    coords = np.array([coord_x, coord_y, coord_z])
    warped = warp(img, coords)
    return warped


def np_warp_2D(img, flow):
    flow = flow.astype('float32')
    height, width = np.shape(img)[0], np.shape(img)[1]
    posx, posy = np.mgrid[:height, :width]
    # flow=np.reshape(flow, [-1, 3])
    vx = flow[:, :, 1]  # to make it consistent as in matlab, ux in python is uy in matlab
    vy = flow[:, :, 0]
    coord_x = posx + vx
    coord_y = posy + vy
    coords = np.array([coord_x, coord_y])
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        img_real = np.real(img).astype('float32')
        img_imag = np.imag(img).astype('float32')
        warped_real = warp(img_real, coords, order=1)
        warped_imag = warp(img_imag, coords, order=1)
        warped = warped_real + 1j*warped_imag
    else:
        img = img.astype('float32')
        warped = warp(img, coords, order=1)  # order=1 for bi-linear

    return warped

