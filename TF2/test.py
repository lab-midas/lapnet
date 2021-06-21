import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import time
from core.flow_util import flow_to_color_np
from core.util import load_mat_file
from core.Warp_assessment3D import warp_assessment3D
from core.image_warp import np_warp_2D, np_warp_3D
from core.undersample.sampling_center import sampleCenter
from core.undersample.retrospective_radial import subsample_radial
from core.cropping import arr2kspace, crop2D_FixPts
from core.undersample.sampling import generate_mask
from preprocess.processing import pos_generation_2D, get_slice_info_from_ods_file, flow_variation, select_2D_Data


def save_img(result, file_path, format='png'):
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.axis('off')
    if len(result.shape) == 2:
        plt.imshow(result, cmap="gray")
    else:
        plt.imshow(result)
    fig.savefig(file_path + '.' + format)
    plt.close()


def show_results(results, name):
    fig, ax = plt.subplots(3, 3, figsize=(14, 14))
    plt.axis('off')
    ax[0][0].imshow(np.abs(results['img_ref']), cmap='gray')
    ax[0][0].set_title('Ref Img')
    ax[0][0].axis('off')
    ax[0][1].imshow(np.abs(results['img_mov']), cmap='gray')
    ax[0][1].set_title('Moving Img')
    ax[0][1].axis('off')
    fig.delaxes(ax[0, 2])

    ax[1][0].imshow(np.abs(results['mov_corr']), cmap='gray')
    ax[1][0].set_title('Moving Corrected')
    ax[1][0].axis('off')
    ax[1][1].imshow(results['color_flow_pred'])
    ax[1][1].set_title('Flow Pred')
    ax[1][1].axis('off')
    ax[1][2].imshow(results['color_flow_gt'])
    ax[1][2].set_title('Flow GT')
    ax[1][2].axis('off')

    ax[2][0].imshow(np.abs(results['err_pred']), cmap='gray')
    ax[2][0].set_title('Warped error')
    ax[2][0].axis('off')
    ax[2][1].imshow(np.abs(results['err_orig']), cmap='gray')
    ax[2][1].set_title('Original Error')
    ax[2][1].axis('off')
    ax[2][2].imshow(np.abs(results['err_gt']), cmap='gray')
    ax[2][2].set_title('GT Error')
    ax[2][2].axis('off')
    plt.show()
    plt.savefig(name + ".png", bbox_inches='tight')


def write_info_in_txt(ID, path, acc, EAE, EPE):
    Subjects_file = open(f'{path}/US{acc}_subject_info.txt', "a")
    EPE_file = open(f'{path}/US{acc}_EPE_loss.txt', "a")
    EAE_file = open(f'{path}/US{acc}_EAE_loss.txt', "a")
    Subjects_file.write(ID + "\n")
    EPE_file.write(str(EPE) + "\n")
    EAE_file.write(str(EAE) + "\n")
    Subjects_file.close()
    EPE_file.close()
    EAE_file.close()


def get_mat_files_EPE_EAE(save_path, ods_file_path, test_data_path, list_ID):
    slices = get_slice_info_from_ods_file(ods_file_path)
    list_us = np.arange(1, 31, 1)
    list_reg_type = ['elastix', 'LAP']
    # list_ID = ['patient_004', 'patient_035', 'patient_036', 'volunteer_06_la', 'volunteer_12_hs']
    for type in list_reg_type:
        for acc in list_us:
            list_paths = []
            for ID in list_ID:
                list_paths += [f'{test_data_path}/{ID}/{type}_{ID}_acc_{acc}.mat']
                for path in list_paths:
                    FlowPath = '/mnt/data/rawdata/MoCo/LAPNet/resp/LAP/'
                    data = sio.loadmat(FlowPath + ID + '.mat')
                    ux = np.asarray(data['ux'], dtype=np.float32)
                    uy = np.asarray(data['uy'], dtype=np.float32)
                    flow_gt_all = np.stack((ux, uy), axis=-1)
                    data = sio.loadmat(path)
                    ux = np.asarray(data['ux'], dtype=np.float32)
                    uy = np.asarray(data['uy'], dtype=np.float32)
                    flow_pred_all = np.stack((ux, uy), axis=-1)
                    ID_slices = slices[ID]
                    for slice in ID_slices:
                        flow_pred = flow_pred_all[:, :, slice, :]
                        flow_gt = flow_gt_all[:, :, slice, :]
                        u_GT = (flow_gt[..., 0], flow_gt[..., 1])  # tuple
                        u_est = (flow_pred[..., 0], flow_pred[..., 1])  # tuple
                        OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
                        error_data_pred = warp_assessment3D(u_GT, u_est, OF_index)

                        final_loss = error_data_pred['Abs_Error_mean']
                        final_loss_angel = error_data_pred['Angle_Error_Mean']
                        write_info_in_txt(f'{ID}_{slice}', f'{save_path}/{type}', acc, final_loss_angel, final_loss)


def save_mean_text_files(txt_path, saving_path):
    files = [txt_path + f for f in os.listdir(txt_path) if not '_info.txt' in f]
    mean_file = open(f'{saving_path}/mean_loss.txt', "a")
    for path in files:
        average, sdv = mean_txt(path)
        ID = os.path.splitext(os.path.basename(path))[0]
        mean_file.write(f'{ID},')
        mean_file.write(f'{average},')
        mean_file.write(f'{sdv}' + "\n")
    mean_file.close()


def mean_txt(path):
    infile = open(path, 'r')
    contents = infile.read().strip().split()
    amount = 0
    for num in contents:
        amount += float(num)
    average = amount / len(contents)

    results = list(map(float, contents))
    sdv = np.std(results)

    infile.close()
    print('mean:', average)
    print('sdv: ', sdv)
    return average, sdv


def eval_cropping(model, experiment_setup):
    weights_path = experiment_setup['weights']
    model.load_weights(weights_path)

    path = experiment_setup['path_data']
    US_acc = experiment_setup['us']
    name = experiment_setup['path_data'].split('/')[-1].split('.')[0]
    slice_info = experiment_setup['slice_info']
    mask_type = experiment_setup['mask_type']
    aug_type = experiment_setup['aug_type']
    savingfile = experiment_setup['save_path']
    experiment_setup['selected_slices'] = slice_info[name]
    try:
        f = load_mat_file(path)
    except:
        try:
            f = np.load(path)
        except ImportError:
            print("Wrong Data Format")

    ref = np.asarray(f['dFixed'], dtype=np.float32)
    ux = np.asarray(f['ux'], dtype=np.float32)  # ux for warp
    uy = np.asarray(f['uy'], dtype=np.float32)
    uz = np.asarray(f['uz'], dtype=np.float32)
    u = np.stack((ux, uy, uz), axis=-1)
    u = flow_variation(ux, u, aug_type)
    mov = np_warp_3D(ref, u)

    if US_acc > 1:
        if mask_type == 'radial':
            ref_mov = np.stack((ref, mov), axis=-1)
            ref_mov = np.ascontiguousarray(ref_mov)
            ref_mov_downsampled = subsample_radial(ref_mov, US_acc, None)
            ref = ref_mov_downsampled[..., 0]
            mov = ref_mov_downsampled[..., 1]
        else:
            if mask_type == 'drUS':
                mask = np.transpose(generate_mask(acc=US_acc, size_y=256, nRep=4), (2, 1, 0))
            elif mask_type == 'crUS':
                mask = sampleCenter(1 / US_acc * 100, 256, 72)
                mask = np.array([mask, ] * 4, dtype=np.float32)
            k_dset = np.multiply(np.fft.fftn(ref), np.fft.ifftshift(mask[0, ...]))
            k_warped_dset = np.multiply(np.fft.fftn(mov), np.fft.ifftshift(mask[3, ...]))
            ref = (np.fft.ifftn(k_dset)).real
            mov = (np.fft.ifftn(k_warped_dset)).real

    ind = experiment_setup['slice_num']
    slice = experiment_setup['selected_slices'][ind]

    Imgs = select_2D_Data(ref, mov, u, slice, 'coronal')
    Imgs = np.asarray(Imgs, dtype=np.float32)
    Imgs = Imgs[np.newaxis, ...]
    radius = int((experiment_setup['slice_size'] - 1) / 2)
    if experiment_setup['padding']:
        Imgs = np.pad(Imgs, ((0, 0), (radius, radius), (radius, radius), (0, 0)), constant_values=0)
    x_dim, y_dim = np.shape(Imgs)[1:3]
    pos = pos_generation_2D(intervall=[[0, x_dim - experiment_setup['slice_size'] + 1],
                                       [0, y_dim - experiment_setup['slice_size'] + 1]], stride=experiment_setup['slice_stride'])

    Imgs_cp = crop2D_FixPts(Imgs, crop_size=experiment_setup['crop_size'], box_num=np.shape(pos)[1], pos=pos)
    Imgs_cp = arr2kspace(Imgs_cp[..., :2])
    # flow = Imgs_cp[:, radius, radius, 4:6]

    flow_pixel = model.predict(Imgs_cp)

    im1 = Imgs[..., 0]
    im2 = Imgs[..., 1]
    flow_orig = Imgs[..., 2:4]
    pos = np.transpose(pos)
    save_path = f'{savingfile}/{name}_{US_acc}'
    eval(flow_pixel, im1, im2, flow_orig, experiment_setup, pos, save_path=save_path)


def eval_tapering(model, experiment_setup, dimensionality):
    weights_path = experiment_setup['weights']
    model.load_weights(weights_path)

    savingfile = experiment_setup['save_path']
    US_acc = experiment_setup['acc']
    name = experiment_setup['path_data'].split('/')[-1].split('.')[0]
    dataID = experiment_setup['path_data']
    data = np.load(dataID)
    batches_cp = data['k_tapered']
    im1 = data['k_ref']
    im2 = data['k_mov']
    flow_orig = data['flow_full']

    height, width = np.shape(im1)
    x_dim = height + experiment_setup['slice_size'] - 1
    y_dim = width + experiment_setup['slice_size'] - 1

    pos = pos_generation_2D(intervall=[[0, x_dim - experiment_setup['slice_size'] + 1],
                                       [0, y_dim - experiment_setup['slice_size'] + 1]], stride=experiment_setup['slice_stride'])
    pos = np.transpose(pos)

    if dimensionality == '2D':
        flow_pixel = model.predict(batches_cp[..., 0])
    else:
        flow_pixel = model.predict((batches_cp[..., 0], batches_cp[..., 1]))

    save_path = f'{savingfile}/{name}_{US_acc}'
    eval(flow_pixel, im1, im2, flow_orig, experiment_setup,dimensionality, pos, save_path=save_path)


def eval(flow_pixel, im1, im2, flow_gt, config, dim, pos, save_path=None, save_txt=False):
    smooth_wind_size = config['smooth_wind_size']
    batch_size = config['batch_size']
    US_acc = config['acc']
    height, width = im1.shape

    if dim == '3D':
        direction = config['direction']
        index1 = 0
        index2 = 1
        if direction == 'sagittal':
            index2 = 2
        if direction == 'axial':
            index1 = 1
            index2 = 2
        flow_pixel = np.stack((flow_pixel[:, index1], flow_pixel[:, index2]), axis=-1)

    flow_raw = np.zeros((height, width, 2), dtype=np.float32)
    time_start = time.time()
    smooth_radius = int((smooth_wind_size - 1) / 2)
    counter_mask = np.zeros((height, width, 2), dtype=np.float32)

    for i in range(int(np.floor(len(pos) / batch_size)) + 1):
        flow_pixel_tmp = flow_pixel[batch_size * i:batch_size * i + batch_size, :]
        local_pos = pos[batch_size * i:batch_size * i + batch_size, :]
        for j in range(len(local_pos)):
            lower_bound_x = max(0, local_pos[j, 0] - smooth_radius)
            upper_bound_x = min(height, local_pos[j, 0] + smooth_radius + 1)
            lower_bound_y = max(0, local_pos[j, 1] - smooth_radius)
            upper_bound_y = min(width, local_pos[j, 1] + smooth_radius + 1)
            flow_raw[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y, :] += flow_pixel_tmp[j, :]
            counter_mask[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y, :] += 1
    flow_final = flow_raw / counter_mask

    time_end1 = time.time()
    print('time cost: {}s'.format(time_end1 - time_start))
    im1_pred = np_warp_2D(im2, -flow_final)

    im_error = im1 - im2
    im_error_pred = im1 - im1_pred

    # warped error of GT
    im1_gt = np_warp_2D(im2, -flow_gt)
    im1_error_gt = im1 - im1_gt

    u_GT = (flow_gt[..., 0], flow_gt[..., 1])  # tuple
    u_est = (flow_final[..., 0], flow_final[..., 1])  # tuple
    OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
    error_data_pred = warp_assessment3D(u_GT, u_est, OF_index)

    size_mtx = np.shape(flow_gt[..., 0])
    u_GT = (np.zeros(size_mtx, dtype=np.float32), np.zeros(size_mtx, dtype=np.float32))  # tuple
    u_est = (flow_gt[..., 0], flow_gt[..., 1])  # tuple
    OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
    error_data_gt = warp_assessment3D(u_GT, u_est, OF_index)

    final_loss_orig = error_data_gt['Abs_Error_mean']
    final_loss = error_data_pred['Abs_Error_mean']
    final_loss_orig_angel = error_data_gt['Angle_Error_Mean']
    final_loss_angel = error_data_pred['Angle_Error_Mean']

    color_flow_final = flow_to_color_np(flow_final, convert_to_bgr=False)
    color_flow_gt = flow_to_color_np(flow_gt, convert_to_bgr=False)

    compare_the_flow = False
    if compare_the_flow:
        # compare the raw and smoothed flow
        fig, ax = plt.subplots(1, 3, figsize=(8, 4))
        ax[0].imshow(flow_gt)  # ref
        ax[0].set_title('Flow GT')
        ax[1].imshow(flow_raw)  # mov
        ax[1].set_title('Flow Pred Raw')
        ax[2].imshow(flow_final)
        ax[2].set_title('Flow Pred Smooth')
        plt.show()

    results = dict()
    results['img_ref'] = im1
    results['img_mov'] = im2
    results['mov_corr'] = im1_pred
    results['color_flow_pred'] = color_flow_final
    results['color_flow_gt'] = color_flow_gt
    results['err_pred'] = im_error_pred
    results['err_orig'] = im_error
    results['err_gt'] = im1_error_gt
    results['flow_pred'] = flow_final
    results['flow_gt'] = flow_gt
    results['loss_pred'] = final_loss
    results['loss_orig'] = final_loss_orig
    results['loss_ang_pred'] = final_loss_angel
    results['loss_ang_orig'] = final_loss_orig_angel
    print('EPE: ', '{:.4f}'.format(final_loss))
    print('EAE: ', '{:.4f}'.format(final_loss_angel))
    show_results(results, save_path)
    if save_txt:
        name = config['path_data'].split('/')[-1].split('.')[0]
        write_info_in_txt(name, save_path, US_acc, final_loss_angel, final_loss)
