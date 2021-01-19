import numpy as np


def kwarp(k, coo):

    return 0

def np_warp_2D(k, flow):
    k = k.astype('float32')
    flow = flow.astype('float32')
    height, width = np.shape(k)[0], np.shape(k)[1]
    posx, posy = np.mgrid[:height, :width]
    # flow=np.reshape(flow, [-1, 3])
    vx = flow[:, :, 1]  # to make it consistent as in matlab, ux in python is uy in matlab
    vy = flow[:, :, 0]

    coord_x = posx + vx
    coord_y = posy + vy
    coords = np.array([coord_x, coord_y])
    warped = kwarp(k, coords)
    return warped