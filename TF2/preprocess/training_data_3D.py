from random import shuffle
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from core.tapering import taper2D
from preprocess.processing import load_data_3D, select_2D_Data, get_maxmin_info_from_ods_file


def save_3D_LAPNet_data_as_npz(data_setup):
    """
    Preprocess data of LAPNet and store them in .npz files
    """
    #settings
    coronalInfo = data_setup['slice_info_coronal']
    sagittalInfo = data_setup['slice_info_sagittal']
    axialInfo = data_setup['slice_info_axial']
    workers = data_setup['num_workers']
    saving_dir = data_setup['saving_dir']
    box_num = data_setup['box_num']
    subjectsIDs = data_setup['subjectsIDs']
    num_subject_us = data_setup['num_subject_us']
    ImgPath = data_setup['ImgPath']
    FlowPath = data_setup['FlowPath']
    mask_Flow = data_setup['mask_Flow']
    normalize = data_setup['normalized_img']

    # read subjects IDs
    infile = open(subjectsIDs, 'r')
    contents = infile.read().strip().split()
    data_paths = [f for f in contents]
    shuffle(data_paths)
    infile.close()
    # read layers numbers
    slice_info_coronal = get_maxmin_info_from_ods_file(coronalInfo)
    slice_info_sagittal = get_maxmin_info_from_ods_file(sagittalInfo)
    slice_info_axial = get_maxmin_info_from_ods_file(axialInfo)

    print('start 3D training data creation ...')
    list_aug = ['real', 'smooth', 'real_x_smooth']
    # create aug file if not existent
    for aug_type in list_aug:
        saving_dir_tapered = f'{saving_dir}/{aug_type}'
        if not os.path.exists(saving_dir_tapered):
            os.makedirs(saving_dir_tapered)
        print(f'start {aug_type} data creation')
        part_pool_func = partial(create_3D_patches,
                                 saving_dir=saving_dir_tapered,
                                 slice_info_sagittal=slice_info_sagittal,
                                 slice_info_axial=slice_info_axial,
                                 slice_info_coronal=slice_info_coronal,
                                 aug_type=aug_type,
                                 ImgPath=ImgPath,
                                 FlowPath=FlowPath,
                                 box_num=box_num,
                                 normalize=normalize,
                                 masking=mask_Flow,
                                 num_subject_us=num_subject_us)

        pool = mp.Pool(workers)
        pool.map(part_pool_func, data_paths)
        pool.close()
        pool.join()
    print('Creating training Dataset done ... ')


def create_3D_patches(ID, saving_dir, aug_type, ImgPath, FlowPath, box_num, slice_info_coronal, slice_info_sagittal,
                      slice_info_axial, normalize, masking, num_subject_us):
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

        # slicing
        coronal_slice_info = slice_info_coronal[ID]
        axial_slice_info = slice_info_axial[ID]
        sagittal_slice_info = slice_info_sagittal[ID]

        for z_dim in range(int(coronal_slice_info[0]), int(coronal_slice_info[1])):
            save_3D_patches_along_depth(ID, ref_3D, mov_3D, u_3D, us_rate, z_dim, box_num, saving_dir,
                                        axial_slice_info, sagittal_slice_info)
    print(f'{ID} {aug_type} done')


def save_3D_patches_along_depth(ID, ref_3D, mov_3D, u_3D, us_rate,
                                z_dim, box_num, saving_dir_tapered,
                                axial_slice_info, sagittal_slice_info, crop_size=33):
    offset = int((crop_size - 1) / 2)
    z_pos = z_dim

    x_pos_list = np.random.randint(int(axial_slice_info[0]), int(axial_slice_info[1]) - crop_size
                                   + 1, box_num)
    y_pos_list = np.random.randint(int(sagittal_slice_info[0]), int(sagittal_slice_info[1]) -
                                   crop_size + 1, box_num)

    slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, z_pos, 'coronal')
    k_ref_coronal = slice_data[:, :, 0]
    k_mov_coronal = slice_data[:, :, 1]

    u_3D_padded = np.pad(u_3D, ((offset, offset), (offset, offset), (offset, offset), (0, 0)), constant_values=0)

    for num in range(box_num):
        train_kspace = np.zeros((crop_size, crop_size, 4, 3), dtype=np.float32)
        npz_saving_path_box = f'{saving_dir_tapered}/{ID}_acc{us_rate}_z_{z_pos}_num_{num}.npz'

        # coronal data
        x_pos = x_pos_list[num]
        y_pos = y_pos_list[num]

        train_kspace[:, :, :2, 0], train_kspace[:, :, 2:4, 0] = taper2D(k_ref_coronal,
                                                                        k_mov_coronal,
                                                                        x_pos,
                                                                        y_pos,
                                                                        crop_size=crop_size)
        # sagittal data
        slice_data_s = select_2D_Data(ref_3D, mov_3D, u_3D, y_pos, 'sagittal')
        k_ref_sagittal = slice_data_s[:, :, 0]
        k_mov_sagittal = slice_data_s[:, :, 1]
        ref_s = np.pad(k_ref_sagittal, (offset, offset), constant_values=0)
        mov_s = np.pad(k_mov_sagittal, (offset, offset), constant_values=0)

        train_kspace[:, :, :2, 1], train_kspace[:, :, 2:4, 1] = taper2D(ref_s,
                                                                        mov_s,
                                                                        x_pos + offset,
                                                                        z_pos + offset,
                                                                        crop_size=crop_size)
        # axial data
        slice_data_a = select_2D_Data(ref_3D, mov_3D, u_3D, x_pos, 'axial')
        k_ref_axial = slice_data_a[:, :, 0]
        k_mov_axial = slice_data_a[:, :, 1]
        ref_a = np.pad(k_ref_axial, (offset, offset), constant_values=0)
        mov_a = np.pad(k_mov_axial, (offset, offset), constant_values=0)
        train_kspace[:, :, :2, 2], train_kspace[:, :, 2:4, 2] = taper2D(ref_a,
                                                                        mov_a,
                                                                        y_pos + offset,
                                                                        z_pos + offset,
                                                                        crop_size=crop_size)

        # 3D flow
        train_flow = u_3D_padded[x_pos + offset, y_pos + offset, z_pos + offset, :]

        # save input
        np.savez(npz_saving_path_box,
                 k_space=train_kspace,
                 flow=train_flow)
