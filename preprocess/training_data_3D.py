import numpy as np
import os
import multiprocessing as mp
from functools import partial

import yaml
import glob

from util import get_maxmin_info_from_ods_file, read_fnames
from processing import respiratoryDataset, taper_selected_slice


def save_3d_data(args):
    """
    Preprocess data of LAPNet and store them in .npz files
    """

    f_names = read_fnames(args.fnames_path)
    print(len(f_names), 'f_names data will be created')
    print('start 2D training data creation ...')

    # read layers numbers
    slice_info_coronal = get_maxmin_info_from_ods_file(args.coronalInfo)
    slice_info_sagittal = get_maxmin_info_from_ods_file(args.sagittalInfo)
    slice_info_axial = get_maxmin_info_from_ods_file(args.axialInfo)

    print('start 3D training data creation ...')
    list_aug = ['real', 'smooth', 'real_x_smooth']
    # create aug file if not existent
    for aug_type in list_aug:
        saving_dir_tapered = f'{args.saving_dir}/{aug_type}'
        if not os.path.exists(saving_dir_tapered):
            os.makedirs(saving_dir_tapered)
        print(f'start {aug_type} data creation')
        part_pool_func = partial(create_3D_patches,
                                 aug_type=aug_type,
                                 saving_dir=saving_dir_tapered,
                                 slice_info_sagittal=slice_info_sagittal,
                                 slice_info_axial=slice_info_axial,
                                 slice_info_coronal=slice_info_coronal,
                                 args=args)

        pool = mp.Pool(args.workers)
        pool.map(part_pool_func, f_names)
        pool.close()
        pool.join()
    print('Creating training Dataset done ... ')


def create_3D_patches(ID, saving_dir, aug_type, slice_info_coronal, slice_info_sagittal,
                      slice_info_axial, args):
    list_us = np.arange(2, 31, 2)
    list_us[0] = 1
    np.random.seed()
    us_rate_list = np.random.RandomState().choice(list_us, size=args.num_subject_us, replace=False)
    dataset = respiratoryDataset(args)

    for us_rate in us_rate_list:
        print(f'{ID} {aug_type} acc {us_rate} start')
        ref_full, mov_full, ref_3D, mov_3D, u_3D, acc = dataset.read_respiratory_data(ID, aug_type, us_rate)

        # slicing
        coronal_slice_info = slice_info_coronal[ID]
        axial_slice_info = slice_info_axial[ID]
        sagittal_slice_info = slice_info_sagittal[ID]

        ID_path = f'{saving_dir}/{ID}*'
        list_ID = glob.glob(ID_path)
        print(len(list_ID), 'the created samples of this', ID)
        for z_dim in range(int(coronal_slice_info[0]), int(coronal_slice_info[1])):
            save_3D_patches_along_depth(ID, ref_3D, mov_3D, ref_full, mov_full,
                                        u_3D,
                                        us_rate, z_dim, args.box_num, saving_dir,
                                        axial_slice_info, sagittal_slice_info)
    print(f'{ID} {aug_type} done')


def save_3D_patches_along_depth(ID, ref_3D, mov_3D, ref_3D_full, mov_3D_full, u_3D, us_rate,
                                z_dim, box_num, saving_dir_tapered,
                                axial_slice_info, sagittal_slice_info, crop_size=33):
    offset = int((crop_size - 1) / 2)
    z_pos = z_dim

    x_pos_list = np.random.randint(int(axial_slice_info[0]), int(axial_slice_info[1]) - crop_size
                                   + 1, box_num)
    y_pos_list = np.random.randint(int(sagittal_slice_info[0]), int(sagittal_slice_info[1]) -
                                   crop_size + 1, box_num)

    u_3D_padded = np.pad(u_3D, ((offset, offset), (offset, offset), (offset, offset), (0, 0)), constant_values=0)

    for num in range(box_num):
        us_kspace = np.zeros((crop_size, crop_size, 4, 3), dtype=np.float32)
        fully_sampled_kspace = np.zeros((crop_size, crop_size, 4, 3), dtype=np.float32)
        npz_saving_path_box = f'{saving_dir_tapered}/{ID}_acc{us_rate}_z_{z_pos}_num_{num}.npz'

        # coronal data
        x_pos = x_pos_list[num]
        y_pos = y_pos_list[num]

        us_kspace[:, :, :2, 0], us_kspace[:, :, 2:4, 0] = taper_selected_slice(ref_3D, mov_3D, u_3D,
                                                                               [x_pos, y_pos, z_pos])
        fully_sampled_kspace[:, :, :2, 0], fully_sampled_kspace[:, :, 2:4, 0] = taper_selected_slice(ref_3D_full,
                                                                                                     mov_3D_full, u_3D,
                                                                                                     [x_pos, y_pos,
                                                                                                      z_pos])

        # sagittal data
        us_kspace[:, :, :2, 1], us_kspace[:, :, 2:4, 1] = taper_selected_slice(ref_3D, mov_3D, u_3D,
                                                                               [x_pos + offset, y_pos, z_pos + offset],
                                                                               'sagittal', offset)
        fully_sampled_kspace[:, :, :2, 1], fully_sampled_kspace[:, :, 2:4, 1] = taper_selected_slice(ref_3D_full,
                                                                                                     mov_3D_full, u_3D,
                                                                                                     [x_pos + offset,
                                                                                                      y_pos,
                                                                                                      z_pos + offset],
                                                                                                     'sagittal', offset)

        # axial data
        us_kspace[:, :, :2, 2], us_kspace[:, :, 2:4, 2] = taper_selected_slice(ref_3D, mov_3D, u_3D,
                                                                               [x_pos, y_pos + offset, z_pos + offset],
                                                                               'axial', offset)
        fully_sampled_kspace[:, :, :2, 2], fully_sampled_kspace[:, :, 2:4, 2] = taper_selected_slice(ref_3D_full,
                                                                                                     mov_3D_full, u_3D,
                                                                                                     [x_pos,
                                                                                                      y_pos + offset,
                                                                                                      z_pos + offset],
                                                                                                     'axial', offset)

        # 3D flow
        train_flow = u_3D_padded[x_pos + offset, y_pos + offset, z_pos + offset, :]

        # save input
        np.savez(npz_saving_path_box,
                 k_space=us_kspace,
                 k_full=fully_sampled_kspace,
                 flow=train_flow)


if __name__ == '__main__':
    with open("/home/studghoul1/lapnet/lapnet/TF2/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    data_setup = data_loaded['Train']['training_data']
    save_3d_data(data_setup)
