import numpy as np
import os
import glob
import yaml

import multiprocessing as mp
from functools import partial

from processing import respiratoryDataset, select_2d_data, cineDataset, save_patches_along_depth
from util import Map, read_fnames, get_maxmin_info_from_ods_file


def save_respiratory_data(args):
    """
    Preprocess data of LAPNet and store them in .npz files
    """
    f_names = read_fnames(args.fnames_path)
    print(len(f_names), 'f_names data will be created')
    print('start 2D training data creation ...')

    # create aug file if not existent
    part_pool_func = partial(create_resp_patches,
                             args=args)
    pool = mp.Pool(args.workers)
    pool.map(part_pool_func, f_names)
    pool.close()
    pool.join()
    print('Creating training Dataset done ... ')


def save_cardiac_data(args):
    f_names = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f'{args.flow_path}/*.mat')]
    print(len(f_names), 'f_names data will be created')
    print('start 2D training data creation ...')

    # create aug file if not existent
    part_pool_func = partial(create_cine_patches,
                             args=args)

    us_list = [np.random.RandomState().choice(args.list_us, size=args.num_subject_us, replace=False) for ID in f_names]

    list_var = [(ID, us) for ID in f_names for us in us_list]

    pool = mp.Pool(args.workers)
    pool.starmap(part_pool_func, list_var)
    pool.close()
    pool.join()
    print('Creating training Dataset done ... ')


def create_resp_patches(ID, args):
    # slicing
    slice_info = get_maxmin_info_from_ods_file(args.info_file)
    ID_slice_info = slice_info[ID]
    dataset = respiratoryDataset(args)

    for aug_type in args.list_aug_types:
        saving_dir_aug = f'{args.saving_dir}/{aug_type}'
        os.makedirs(saving_dir_aug, exist_ok=True)
        np.random.seed()

        num_subject_us = args.num_subject_us

        if aug_type == 'real':
            num_subject_us = int(num_subject_us / 2)
        us_rate_list = np.random.RandomState().choice(args.list_us, size=num_subject_us, replace=False)
        print(us_rate_list, 'accelerations will be created for', ID, 'with the augmentation', aug_type)
        for acc in us_rate_list:
            ref_full, mov_full, ref_3D, mov_3D, u_3D, acc = dataset.read_respiratory_data(ID, aug_type, acc)
            for z_dim in range(int(ID_slice_info[0]), int(ID_slice_info[1])):
                ref_img, mov_img, u = select_2d_data(ref_3D, mov_3D, u_3D, z_dim, args.direction)
                ref_img_full, mov_img_full, _ = select_2d_data(ref_full, mov_full, u_3D, z_dim, args.direction)
                saving_path = f'{saving_dir_aug}/{ID}_acc{acc}_slice{z_dim}'
                save_patches_along_depth(ref_img, mov_img, u, args.box_num, saving_path,
                                         taper_size=args.taper_size,
                                         ref_full=ref_img_full,
                                         mov_full=mov_img_full
                                         )


def create_cine_patches(f_name, acc, args):
    # slicing
    slice_info = get_maxmin_info_from_ods_file(args.info_file)
    ID_slice_info = slice_info[f_name]
    dataset = cineDataset(args)

    for aug_type in args.list_aug_types:
        saving_dir_aug = f'{args.saving_dir}/{aug_type}'
        os.makedirs(saving_dir_aug, exist_ok=True)
        np.random.seed()
        num_subject_us = args.num_subject_us

        if aug_type == 'real':
            num_subject_us = int(num_subject_us / 2)
        us_rate_list = np.random.RandomState().choice(args.list_us, size=num_subject_us, replace=False)
        print(us_rate_list, 'accelerations will be created for', f_name, 'with the augmentation', aug_type)
        for s in range(int(ID_slice_info[0]), int(ID_slice_info[1])):
            ref, mov, sub_ref, sub_mov, u = dataset.read_cardiac_data(f_name, aug_type, acc, s)
            saving_dir = f'{saving_dir_aug}/{f_name}_acc{acc}_slice{s}'
            save_patches_along_depth(sub_ref, sub_mov, u, args.box_num, saving_dir, ref, mov)


if __name__ == '__main__':
    with open("/home/studghoul1/lapnet/lapnet/TF2/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    data_setup = data_loaded['Train']['training_data']
    save_respiratory_data(Map(data_setup))
