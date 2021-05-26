from processing import load_data_3D, select_2D_Data, pos_generation_2D
import numpy as np
from TF2.core.tapering import taper2D


def create_sagittal_data(dataID, us_acc, slice_num, save_dir, tapering_size=33, crop_stride=2, normalized=False,
                         ImgPath='/mnt/data/rawdata/MoCo/LAPNet/resp/motion_data',
                         FlowPath='/mnt/data/rawdata/MoCo/LAPNet/resp/LAP'):
    ref_3D, mov_3D, u_3D, us_rate = load_data_3D(dataID, ImgPath, FlowPath, 'real', us_acc, 'drUS', normalized)
    slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, slice_num, 'sagittal')

    radius = int((tapering_size - 1) / 2)
    k_ref_sagittal = slice_data[:, :, 0]
    k_mov_sagittal = slice_data[:, :, 1]
    ref_s = np.pad(k_ref_sagittal, (radius, radius), constant_values=0)
    mov_s = np.pad(k_mov_sagittal, (radius, radius), constant_values=0)
    u_3D = np.pad(u_3D, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

    # extract data
    k_u = np.real(slice_data[..., 2:])

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
        slice_data_c = select_2D_Data(ref_3D, mov_3D, u_3D, z_pos, 'coronal')
        k_ref_coronal = slice_data_c[:, :, 0]
        k_mov_coronal = slice_data_c[:, :, 1]
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
        slice_data_a = select_2D_Data(ref_3D, mov_3D, u_3D, x_pos, 'axial')
        k_ref_axial = slice_data_a[:, :, 0]
        k_mov_axial = slice_data_a[:, :, 1]
        ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
        mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 2], k_output[num, :, :, 2:4, 2] = taper2D(ref_a,
                                                                          mov_a,
                                                                          y_pos,
                                                                          z_pos)

        # 3D flow
        flow_output[num, ...] = u_3D[x_pos + radius, y_pos + radius, z_pos + radius, :]

    np.savez(f'{save_dir}/sagital_{dataID}_acc{us_acc}_slice{slice_num}',
             k_ref=k_ref_sagittal,
             k_mov=k_mov_sagittal,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)


def create_axial_data(dataID, us_acc, slice_num, save_dir, tapering_size=33, crop_stride=2, normalized=False,
                      ImgPath='/mnt/data/rawdata/MoCo/LAPNet/resp/motion_data',
                      FlowPath='/mnt/data/rawdata/MoCo/LAPNet/resp/LAP'):
    ref_3D, mov_3D, u_3D, us_rate = load_data_3D(dataID, ImgPath, FlowPath, 'real', us_acc, 'drUS', normalized)
    slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, slice_num, 'axial')
    radius = int((tapering_size - 1) / 2)
    k_ref_axial = slice_data[:, :, 0]
    k_mov_axial = slice_data[:, :, 1]
    ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
    mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)

    # extract data
    k_u = np.real(slice_data[..., 2:])
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
        slice_data_c = select_2D_Data(ref_3D, mov_3D, u_3D, z_pos, 'coronal')
        k_ref_coronal = slice_data_c[:, :, 0]
        k_mov_coronal = slice_data_c[:, :, 1]
        ref_c = np.pad(k_ref_coronal, (radius, radius), constant_values=0)
        mov_c = np.pad(k_mov_coronal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 0], k_output[num, :, :, 2:4, 0] = taper2D(ref_c,
                                                                          mov_c,
                                                                          x_pos,
                                                                          y_pos)

        # sagittal data
        slice_data_s = select_2D_Data(ref_3D, mov_3D, u_3D, y_pos, 'sagittal')
        k_ref_sagittal = slice_data_s[:, :, 0]
        k_mov_sagittal = slice_data_s[:, :, 1]
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

    np.savez(f'{save_dir}/axial_{dataID}_acc{us_acc}_slice{slice_num}',
             k_ref=k_ref_axial,
             k_mov=k_mov_axial,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)


def create_coronal_data(dataID, us_acc, slice_num, save_dir, tapering_size=33, crop_stride=2, normalized=False,
                        ImgPath='/mnt/data/rawdata/MoCo/LAPNet/resp/motion_data',
                        FlowPath='/mnt/data/rawdata/MoCo/LAPNet/resp/LAP'):
    ref_3D, mov_3D, u_3D, us_rate = load_data_3D(dataID, ImgPath, FlowPath, 'real', us_acc, 'drUS', normalized)
    slice_data = select_2D_Data(ref_3D, mov_3D, u_3D, slice_num, 'axial')
    radius = int((tapering_size - 1) / 2)
    k_ref_coronal = slice_data[:, :, 0]
    k_mov_coronal = slice_data[:, :, 1]
    ref_c = np.pad(k_ref_coronal, (radius, radius), constant_values=0)
    mov_c = np.pad(k_mov_coronal, (radius, radius), constant_values=0)

    # extract data
    k_u = np.real(slice_data[..., 2:])
    u_3D = np.pad(u_3D, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), constant_values=0)

    x_dim, y_dim = np.shape(ref_c)
    pos = pos_generation_2D(intervall=[[0, x_dim - tapering_size + 1],
                                       [0, y_dim - tapering_size + 1]], stride=crop_stride)
    k_output = np.zeros((pos.shape[1], tapering_size, tapering_size, 4, 3), dtype=np.float32)
    flow_output = np.zeros((pos.shape[1], 3), dtype=np.float32)

    for num in range(pos.shape[1]):
        # print(num)
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
        slice_data_s = select_2D_Data(ref_3D, mov_3D, u_3D, y_pos, 'sagittal')
        k_ref_sagittal = slice_data_s[:, :, 0]
        k_mov_sagittal = slice_data_s[:, :, 1]
        ref_s = np.pad(k_ref_sagittal, (radius, radius), constant_values=0)
        mov_s = np.pad(k_mov_sagittal, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 1], k_output[num, :, :, 2:4, 1] = taper2D(ref_s,
                                                                          mov_s,
                                                                          x_pos,
                                                                          z_pos)
        # axial data
        slice_data_a = select_2D_Data(ref_3D, mov_3D, u_3D, x_pos, 'axial')
        k_ref_axial = slice_data_a[:, :, 0]
        k_mov_axial = slice_data_a[:, :, 1]
        ref_a = np.pad(k_ref_axial, (radius, radius), constant_values=0)
        mov_a = np.pad(k_mov_axial, (radius, radius), constant_values=0)
        k_output[num, :, :, :2, 2], k_output[num, :, :, 2:4, 2] = taper2D(ref_a,
                                                                          mov_a,
                                                                          y_pos,
                                                                          z_pos)

        # 3D flow
        flow_output[num, ...] = u_3D[x_pos + radius, y_pos + radius, z_pos + radius, :]

    np.savez(f'{save_dir}/coronal_{dataID}_acc{us_acc}_slice_{slice_num}',
             k_ref=k_ref_coronal,
             k_mov=k_mov_coronal,
             flow_full=k_u,
             k_tapered=k_output,
             flow_tapered=flow_output)
