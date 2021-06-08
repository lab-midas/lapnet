from TF2.core.tapering import taper2D
import os
from TF2.preprocess.processing import pos_generation_2D, undersample, select_2D_Data, load_data_3D, get_slice_info_from_ods_file
import numpy as np
from random import shuffle
import multiprocessing as mp
from functools import partial


def create_2D_test_dataset(data_setup):
    # settings
    workers = data_setup['num_workers']
    saving_dir = data_setup['saving_dir']
    list_us = data_setup['list_us']
    ImgPath = data_setup['ImgPath']
    FlowPath = data_setup['FlowPath']
    subjectsIDs = data_setup['subjectsIDs']
    mask_type = data_setup['mask_type']
    aug_type = data_setup['aug_type']
    masking = data_setup['mask_Flow']
    info_file = data_setup['slice_info_coronal']
    normalized = data_setup['normalized_img']

    infile = open(subjectsIDs, 'r')
    contents = infile.read().strip().split()
    data_paths = [f for f in contents]
    shuffle(data_paths)
    infile.close()
    print('start test data creation')
    for dataID in data_paths:
        print(f'create {dataID} test data')
        ref_3D, mov_3D, u_3D, us_rate = load_data_3D(dataID, ImgPath, FlowPath, aug_type, 1, mask_type, normalized, masking)
        slice_info = get_slice_info_from_ods_file(info_file)
        slice_info_ID = slice_info[dataID]
        slice_in = slice_info_ID[0]
        part_pool_func = partial(create_test_data,
                                 ID=dataID,
                                 ref_3D=ref_3D,
                                 mov_3D=mov_3D,
                                 u_3D=u_3D,
                                 slice_in=slice_in,
                                 save_dir=saving_dir)
        list_var = [(us, slice_num) for us in list_us for slice_num in slice_info_ID]
        pool = mp.Pool(workers)
        pool.starmap(part_pool_func, list_var)
        pool.close()
        pool.join()
    print('end test data creation')


def create_test_data(us_acc,
                     slice_num,
                     save_dir,
                     ID,
                     ref_3D,
                     mov_3D,
                     u_3D,
                     slice_in,
                     tapering_size=33,
                     crop_stride=2):
    saving_dir_tapered = f'{save_dir}/{us_acc}'
    if not os.path.exists(saving_dir_tapered):
        os.makedirs(saving_dir_tapered)
    slice_idx = slice_num - slice_in
    npz_saving_path = f'{saving_dir_tapered}/{ID}_slice{slice_idx}.npz'
    if not os.path.exists(npz_saving_path):
        ref_3D, mov_3D = undersample(ref_3D, mov_3D, us_acc, 'drUS', False)
        slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, slice_num, 'coronal')
        print(f'{ID}: create slice {slice_idx} for acceleration {us_acc}')
        # extract data
        k_ref = slice_data[:, :, 0]
        k_mov = slice_data[:, :, 1]
        k_u = np.real(slice_data[..., 2:])

        # padding
        radius = int((tapering_size - 1) / 2)
        ref = np.pad(k_ref, (radius, radius), constant_values=0)
        mov = np.pad(k_mov, (radius, radius), constant_values=0)
        u = np.pad(k_u, ((radius, radius), (radius, radius), (0, 0)), constant_values=0)

        x_dim, y_dim = np.shape(ref)
        pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                           [0, y_dim - tapering_size + 1]], stride=crop_stride)
        k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4), dtype=np.float32)
        flow_output = np.zeros((pos.shape[1], 2), dtype=np.float32)

        # Generate tapered data
        for i in range(pos.shape[1]):
            pos_tmp = pos[:, i]
            k_output[i, :, :, :2], k_output[i, :, :, 2:4], flow_output[i, :] = taper2D(ref, mov, pos_tmp[0], pos_tmp[1],
                                                                                       tapering_size,
                                                                                       u=u)
        np.savez(npz_saving_path,
                 k_ref=k_ref,
                 k_mov=k_mov,
                 flow_full=k_u,
                 k_tapered=k_output,
                 flow_tapered=flow_output)
    else:
        print(f'{ID}: acceleration {us_acc} slice index {slice_idx} exists')
