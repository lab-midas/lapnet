import numpy as np
import time
import matplotlib.pyplot as plt
import os
from image_warp import np_warp_2D
from Warp_assessment3D import warp_assessment3D
from core.flow_util import flow_to_color_np
import torch


def read_test_data(name, acc, args):
    dataID = f'{args.data_path}/{acc}/{name}.npz'
    data = np.load(dataID)
    im1 = data['k_ref']
    im2 = data['k_mov']
    flow_orig = data['flow_full']
    batches_cp = data['k_tapered']
    return im1, im2, flow_orig, batches_cp


def pos_generation_2D(intervall, stride):
    """
    :param intervall:
    :param stride:
    :return: 2 x position (x, y)
    """
    x = np.arange(intervall[0][0], intervall[0][1], stride)
    y = np.arange(intervall[1][0], intervall[1][1], stride)
    vx, vy = np.meshgrid(x, y)
    vx = vx.reshape(vx.shape[1] * vx.shape[0])
    vy = vy.reshape(vy.shape[1] * vy.shape[0])
    pos = np.stack((vx, vy), axis=0)
    return pos

def show_results(results, name, args):
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
    # plt.savefig(f'{args.save_path}/{name}.png', bbox_inches='tight')

def generate_pos_eval(args):
    height, width = args.img_size
    x_dim = height + args.patch_size - 1
    y_dim = width + args.patch_size - 1

    pos = pos_generation_2D(intervall=[[0, x_dim - args.patch_size + 1],
                                       [0, y_dim - args.patch_size + 1]],
                            stride=args.stride_size)

    pos = np.transpose(pos)
    return pos

def get_dic_results(im1, im2, flow_gt, flow_final):
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

    return results

def arrange_predicted_flow(flow_pixel,  args):
    pos = generate_pos_eval(args)
    smooth_wind_size = args.smooth_wind_size
    batch_size = args.batch_size
    height, width = args.img_size

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
    return flow_final

def write_info_in_txt(acc, results, args):
    ID = args.path_data.split('/')[-1].split('.')[0]
    EAE = results['loss_ang_pred']
    EPE = results['loss_pred']
    Subjects_file = open(f'{args.save_path}/US{acc}_subject_info.txt', "a")
    EPE_file = open(f'{args.save_path}/US{acc}_EPE_loss.txt', "a")
    EAE_file = open(f'{args.save_path}/US{acc}_EAE_loss.txt', "a")
    Subjects_file.write(ID + "\n")
    EPE_file.write(str(EPE) + "\n")
    EAE_file.write(str(EAE) + "\n")
    Subjects_file.close()
    EPE_file.close()
    EAE_file.close()

def get_sample_results(model, name, acc, args):
    # device configuration
    device = torch.device(args.cuda_name)
    im1, im2, flow_orig, batches_cp = read_test_data(name, acc, args)

    data = torch.from_numpy(np.transpose(batches_cp, (0, 3, 2, 1)))
    flow_pixel = model(data.to(device))
    flow_pixel = flow_pixel.cpu().detach().numpy()

    flow_final = arrange_predicted_flow(flow_pixel, args)
    results = get_dic_results(im1, im2, flow_orig, flow_final)
    print('EPE: ', '{:.4f}'.format(results['loss_pred']))
    print('EAE: ', '{:.4f}'.format(results['loss_ang_pred']))
    show_results(results, name, args)

    if args.save_txt:
        write_info_in_txt(acc, results, args)
    return results

def eval_img(results, show=True, save_path=None):
    fig, ax = plt.subplots(3, 5, figsize=(14, 14))
    plt.axis('off')
    # add images
    for i, data in enumerate(results):
        ax[i][0].imshow(np.abs(data['img_ref']), cmap='gray')
        ax[i][0].set_title('Ref Img')
        ax[i][1].imshow(np.abs(data['img_mov']), cmap='gray')
        ax[i][1].set_title('Moving Img')
        ax[i][2].imshow(np.abs(data['mov_corr']), cmap='gray')
        ax[i][2].set_title('Moving Corrected')
        ax[i][3].imshow(data['color_flow_pred'])
        ax[i][3].set_title('Flow Pred')
        EPE = 'EPE: ' + '{:.4f}'.format(data['loss_pred'])
        EAE = 'EAE: ' + '{:.4f}'.format(data['loss_ang_pred'])
        ax[i][3].text(10, 310, EPE, horizontalalignment='left', fontsize=12, verticalalignment='center')
        ax[i][3].text(10, 275, EAE, horizontalalignment='left', fontsize=12, verticalalignment='center')
    ax[0][4].imshow(results[0]['color_flow_gt'])
    ax[0][4].set_title('Flow LAP masked')
    for i in range(5):
        for j in range(3):
            ax[j][i].axis('off')
            # add US_rates:
    list_us = ['fully sampled', 'acc=8x', 'acc=30x']
    for ind, us_text in enumerate(list_us):
        ax[ind][0].text(-20, 120, us_text, rotation=90, horizontalalignment='center', fontsize=12,
                        verticalalignment='center')
    if save_path:
        plt.savefig(f"{save_path}/eval_img.png", bbox_inches='tight')
        print(f'evaluation figure is saved under {save_path}')
    if show:
        plt.show()



