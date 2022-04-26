import os
from processing import respiratoryDataset, select_2d_data
import numpy as np
import yaml
from functools import partial
from core import taper2D, pos_generation_2D
import multiprocessing as mp
from util import Map


def create_sagittal_data(dataID, us_acc, slice_num, args):
    dataset = respiratoryDataset(args)
    ref_3D, mov_3D, u_3D, us_rate = dataset.read_respiratory_data(dataID, 'real', us_acc)
    k_ref_sagittal, k_mov_sagittal, k_u = select_2d_data(ref_3D, mov_3D, u_3D, slice_num, 'sagittal')

    tapering_size = args.tapering_size
    crop_stride = args.crop_stride
    radius = int((tapering_size - 1) / 2)
    ref_s = np.pad(k_ref_sagittal, (radius, radius), constant_values=0)
    mov_s = np.pad(k_mov_sagittal, (radius, radius), constant_values=0)
    u_3D = np.pad(u_3D, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

    x_dim, z_dim = np.shape(ref_s)
    pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                       [0, z_dim - tapering_size + 1]], stride=crop_stride)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4, 3), dtype=np.float32)
    flow_output = np.zeros((pos.shape[1], 3), dtype=np.float32)
    print(pos.shape)
    for num in range(pos.shape[1]):
        # print(num)
        pos_tmp = pos[:, num]

        # positions
        x_pos = pos_tmp[0]
        y_pos = slice_num + radius
        z_pos = pos_tmp[1]

        # coronal data
        k_ref_coronal, k_mov_coronal, _ = select_2d_data(ref_3D, mov_3D, u_3D, z_pos, 'coronal')
        ref_c = np.pad(k_ref_coronal, (radius, radius), constant_values=0)
        mov_c = np.pad(k_mov_coronal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 0], k_output[num, :, :, 2:4, 0] = taper2D(ref_c,
                                                                          mov_c,
                                                                          x_pos,
                                                                          y_pos)
        # sagittal data
        k_output[num, :, :, :2, 1], k_output[num, :, :, 2:4, 1] = taper2D(ref_s,
                                                                          mov_s,
                                                                          x_pos,
                                                                          z_pos)
        # axial data
        k_ref_axial, k_mov_axial, _ = select_2d_data(ref_3D, mov_3D, u_3D, x_pos, 'axial')
        ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
        mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 2], k_output[num, :, :, 2:4, 2] = taper2D(ref_a,
                                                                          mov_a,
                                                                          y_pos,
                                                                          z_pos)

        # 3D flow
        flow_output[num, ...] = u_3D[x_pos + radius, y_pos + radius, z_pos + radius, :]

    os.makedirs(f'{args.save_dir}/sagittal/{us_acc}', exist_ok=True)

    np.savez(f'{args.save_dir}/sagittal/{us_acc}/{dataID}_slice{slice_num}.npz',
             k_ref=k_ref_sagittal,
             k_mov=k_mov_sagittal,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)


def create_axial_data(dataID, us_acc, slice_num, args):
    dataset = respiratoryDataset(args)
    ref_3D, mov_3D, u_3D, us_rate = dataset.read_respiratory_data(dataID, 'real', us_acc)
    k_ref_axial, k_mov_axial, k_u = select_2d_data(ref_3D, mov_3D, u_3D, slice_num, 'axial')
    tapering_size = args.tapering_size
    crop_stride = args.crop_stride
    radius = int((tapering_size - 1) / 2)
    ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
    mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)

    # extract data
    u_3D = np.pad(u_3D, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

    y_dim, z_dim = np.shape(ref_a)
    pos = pos_generation_2D(intervall=[[0, y_dim - tapering_size + 1],
                                       [0, z_dim - tapering_size + 1]], stride=crop_stride)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4, 3), dtype=np.float32)
    flow_output = np.zeros((pos.shape[1], 3), dtype=np.float32)
    print(pos)

    for num in range(pos.shape[1]):
        pos_tmp = pos[:, num]
        # print(num)

        # positions
        y_pos = pos_tmp[0]
        z_pos = pos_tmp[1]
        x_pos = slice_num + radius

        # coronal data
        k_ref_coronal, k_mov_coronal, _ = select_2d_data(ref_3D, mov_3D, u_3D, z_pos, 'coronal')
        ref_c = np.pad(k_ref_coronal, (radius, radius), constant_values=0)
        mov_c = np.pad(k_mov_coronal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 0], k_output[num, :, :, 2:4, 0] = taper2D(ref_c,
                                                                          mov_c,
                                                                          x_pos,
                                                                          y_pos)

        # sagittal data
        k_ref_sagittal, k_mov_sagittal, _ = select_2d_data(ref_3D, mov_3D, u_3D, y_pos, 'sagittal')
        ref_s = np.pad(k_ref_sagittal, (radius, radius), constant_values=0)
        mov_s = np.pad(k_mov_sagittal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 1], k_output[num, :, :, 2:4, 1] = taper2D(ref_s,
                                                                          mov_s,
                                                                          x_pos,
                                                                          z_pos)
        # axial data
        k_output[num, :, :, :2, 2], k_output[num, :, :, 2:4, 2] = taper2D(ref_a,
                                                                          mov_a,
                                                                          y_pos,
                                                                          z_pos)

        # 3D flow
        flow_output[num, ...] = u_3D[x_pos + radius, y_pos + radius, z_pos + radius, :]

    np.savez(f'{args.save_dir}/axial_{dataID}_acc{us_acc}_slice{slice_num}',
             k_ref=k_ref_axial,
             k_mov=k_mov_axial,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)


def create_coronal_data(dataID, us_acc, slice_num, args):
    dataset = respiratoryDataset(args)
    ref_3D, mov_3D, u_3D, us_rate = dataset.read_respiratory_data(dataID, 'real', us_acc)
    k_ref_coronal, k_mov_coronal, k_u = select_2d_data(ref_3D, mov_3D, u_3D, slice_num, 'coronal')
    tapering_size = args.tapering_size
    crop_stride = args.crop_stride
    radius = int((tapering_size - 1) / 2)
    ref_c = np.pad(k_ref_coronal, (radius, radius), constant_values=0)
    mov_c = np.pad(k_mov_coronal, (radius, radius), constant_values=0)
    # extract data
    u_3D = np.pad(u_3D, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

    x_dim, y_dim = np.shape(ref_c)
    pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                       [0, y_dim - tapering_size + 1]], stride=crop_stride)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4, 3), dtype=np.float32)
    flow_output = np.zeros((pos.shape[1], 3), dtype=np.float32)

    for num in range(pos.shape[1]):
        print(num)
        pos_tmp = pos[:, num]

        # coronal data
        x_pos = pos_tmp[0]
        y_pos = pos_tmp[1]
        z_pos = slice_num + radius
        k_output[num, :, :, :2, 0], k_output[num, :, :, 2:4, 0] = taper2D(ref_c,
                                                                          mov_c,
                                                                          x_pos,
                                                                          y_pos)
        # sagittal data
        k_ref_sagittal, k_mov_sagittal, _ = select_2d_data(ref_3D, mov_3D, u_3D, y_pos, 'sagittal')
        ref_s = np.pad(k_ref_sagittal, (radius, radius), constant_values=0)
        mov_s = np.pad(k_mov_sagittal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 1], k_output[num, :, :, 2:4, 1] = taper2D(ref_s,
                                                                          mov_s,
                                                                          x_pos,
                                                                          z_pos)
        # axial data
        k_ref_axial, k_mov_axial, _ = select_2d_data(ref_3D, mov_3D, u_3D, x_pos, 'axial')
        ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
        mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 2], k_output[num, :, :, 2:4, 2] = taper2D(ref_a,
                                                                          mov_a,
                                                                          y_pos,
                                                                          z_pos)

        # 3D flow
        flow_output[num, ...] = u_3D[x_pos + radius, y_pos + radius, z_pos + radius, :]

    os.makedirs(f'{args.save_dir}/coronal/{us_acc}', exist_ok=True)

    np.savez(f'{args.save_dir}/coronal/{us_acc}/{dataID}_slice{slice_num}.npz',
             k_ref=k_ref_coronal,
             k_mov=k_mov_coronal,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)


def create_ID_data(dataID, args):
    acc_list = [1, 8, 30]
    slice_num_cor = 28
    slice_num_sag = 90

    for us_acc in acc_list:
        create_sagittal_data(dataID, us_acc, slice_num_sag, args)
        create_coronal_data(dataID, us_acc, slice_num_cor, args)


if __name__ == '__main__':
    workers = 5
    dataIDs = ['patient_004', 'patient_035', 'patient_036', 'volunteer_06_la', 'volunteer_12_hs']
    with open("/home/studghoul1/lapnet/lapnet/TF2/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    data_setup = data_loaded['Test']['test_data']
    args = Map(data_setup)
    part_pool_func = partial(create_ID_data, args=args)
    pool = mp.Pool(workers)
    pool.map(part_pool_func, dataIDs)
    pool.close()
    pool.join()
