from random import shuffle
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from ...core.tapering import taper2D
from processing import load_data_3D, select_2D_Data, get_maxmin_info_from_ods_file


def save_2D_LAPNet_data_as_npz(data_setup):
    """
    Preprocess data of LAPNet and store them in .npz files
    """
    # settings
    workers = data_setup['num_workers']
    saving_dir = data_setup['saving_dir']
    box_num = data_setup['box_num']
    subjectsIDs = data_setup['subjectsIDs']
    num_subject_us = data_setup['num_subject_us']
    ImgPath = data_setup['ImgPath']
    FlowPath = data_setup['FlowPath']
    mask_Flow = data_setup['mask_Flow']
    normalize = data_setup['normalized_img']
    sliceInfo = data_setup['slice_info_coronal']
    # read subjects IDs
    infile = open(subjectsIDs, 'r')
    contents = infile.read().strip().split()
    data_paths = [f for f in contents]
    shuffle(data_paths)
    infile.close()

    slice_info = get_maxmin_info_from_ods_file(sliceInfo)
    print('start 2D training data creation ...')
    list_aug = ['real', 'smooth', 'real_x_smooth']
    # create aug file if not existent
    for aug_type in list_aug:
        saving_dir_tapered = f'{saving_dir}/{aug_type}'
        if not os.path.exists(saving_dir_tapered):
            os.makedirs(saving_dir_tapered)
        print(f'start {aug_type} data creation')
        part_pool_func = partial(create_2D_patches,
                                 saving_dir=saving_dir_tapered,
                                 aug_type=aug_type,
                                 ImgPath=ImgPath,
                                 slice_info=slice_info,
                                 FlowPath=FlowPath,
                                 box_num=box_num,
                                 masking=mask_Flow,
                                 normalize=normalize,
                                 num_subject_us=num_subject_us)

        pool = mp.Pool(workers)
        pool.map(part_pool_func, data_paths)
        pool.close()
        pool.join()
    print('Creating training Dataset done ... ')


def create_2D_patches(ID, saving_dir, aug_type, ImgPath, FlowPath, box_num, slice_info, num_subject_us, normalize,
                      masking,
                      direction='coronal'):
    # slicing
    ID_slice_info = slice_info[ID]
    # acceleration list
    list_us = np.arange(0, 31, 2)
    list_us[0] = 1
    np.random.seed()
    us_rate_list = np.random.RandomState().choice(list_us, size=num_subject_us, replace=False)
    for us_rate in us_rate_list:
        print(f'{ID} {aug_type} acc {us_rate} start')
        ref_3D, mov_3D, u_3D, acc = load_data_3D(dataID=ID,
                                                 img_path=ImgPath,
                                                 flow_path=FlowPath,
                                                 aug_type=aug_type,
                                                 us_rate=us_rate,
                                                 mask_type='drUS',
                                                 normalized=normalize,
                                                 masking=masking)

        for z_dim in range(int(ID_slice_info[0]), int(ID_slice_info[1])):
            slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, z_dim, direction)
            ref_img = slice_data[:, :, 0]
            mov_img = slice_data[:, :, 1]
            u = slice_data[:, :, :2]
            save_2D_patches_along_depth(ID, ref_img, mov_img, u, us_rate, z_dim, box_num, saving_dir)
    print(f'{ID} {aug_type} done')


def save_2D_patches_along_depth(ID, ref_img, mov_img, u, us_rate, z_dim, box_num, saving_dir, taper_size=33):
    x_dim, y_dim = ref_img.shape

    train_kspace = np.zeros((box_num, taper_size, taper_size, 4), dtype=np.float32)
    train_flow = np.zeros((box_num, 2), dtype=np.float32)

    x_pos = np.random.randint(0, x_dim - taper_size + 1, box_num)
    y_pos = np.random.randint(0, y_dim - taper_size + 1, box_num)

    for num in range(box_num):
        train_kspace[num, :, :, :2], train_kspace[num, :, :, 2:], train_flow[num, :] = taper2D(ref_img,
                                                                                               mov_img,
                                                                                               x_pos[num],
                                                                                               y_pos[num],
                                                                                               u=u,
                                                                                               crop_size=taper_size)
    np.savez(f'{saving_dir}/{ID}_acc{us_rate}_slice{z_dim}.npz',
             k_space=train_kspace,
             flow=train_flow)
