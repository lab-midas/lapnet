import matplotlib.pyplot as plt
import numpy as np


def increase_brightness(img, value=0.5):
    max = np.max(img)
    value = max * value
    lim = max - value
    img[img > lim] = max
    img[img <= lim] += value
    return img


def eval_plot(results, list_accs, show=True, save_path=None):
    num_plots = 5
    stride = 8
    scale = 60

    fig, ax = plt.subplots(len(results), num_plots, figsize=(16, 10))
    fig.tight_layout()
    # fig.tight_layout(h_pad=1, w_pad=1)
    plt.axis('off')
    # add images
    for i, data in enumerate(results):
        ax[i][0].imshow(increase_brightness(np.abs(data['img_ref'])), cmap='gray')

        ax[i][1].imshow(increase_brightness(np.abs(data['img_mov'])), cmap='gray')

        ax[i][3].imshow(increase_brightness(np.abs(data['mov_corr'])), cmap='gray')

        ax[i][4].imshow(data['color_flow_pred'])

        photo_loss = 'loss:' + '{:.4f}'.format(np.abs(data['err_pred']).mean())
        psnr = 'PSNR: ' + '{:.3f}'.format(data['psnr'].mean())
        ssim = 'SSIM: ' + '{:.4f}'.format(data['ssim'].mean())
        x = 0
        y = 15
        ax[i][4].text(10, 280, photo_loss, horizontalalignment='left', fontsize=16, verticalalignment='center')
        # ax[i][6].text(10, 180+x+y, psnr, horizontalalignment='left',   fontsize=13, verticalalignment='center')
        # ax[i][6].text(10, 180+x+2*y, ssim, horizontalalignment='left', fontsize=13, verticalalignment='center')

        # ax[i][4].imshow(np.abs(data['err_orig'] / 255), cmap='gray')
        #
        # ax[i][5].imshow(np.abs(data['err_pred'] / 255), cmap='gray')

        ax[i][2].imshow(np.abs(data['img_mov']), cmap='gray')
        ax[i][2].axis('off')
        u = data['flow_pred']
        x, y = u.shape[:2]
        gridx, gridy = np.meshgrid(np.arange(0, y, stride), np.arange(0, x, stride))
        ax[i][2].quiver(gridx, gridy, u[0:x:stride, :, :][:, 0:y:stride, :][:, :, 0],
                        -u[0:x:stride, :, :][:, 0:y:stride, :][:, :, 1], color='y', scale_units='inches', scale=scale,
                        headwidth=5)

    font = 16

    # ax[0][5].set_title('Ref - Pred'      , fontsize   = font)
    # ax[0][4].set_title('Ref - Moving'    , fontsize   = font)
    ax[0][0].set_title('Ref Img', fontsize=font)
    ax[0][1].set_title('Moving Img', fontsize=font)
    ax[0][2].set_title('Moving + flow', fontsize=font)
    ax[0][3].set_title('Moving Corrected', fontsize=font)
    ax[0][4].set_title('LAPNet', fontsize=font)

    # ax[0][7].imshow(data['color_flow_gt'])
    # ax[0][7].axis('off')
    # ax[0][7].set_title('LAP')

    for i in range(len(results)):
        for j in range(num_plots):
            ax[i][j].axis('off')
            # add US_rates:
    list_us = [f'x{acc}' for acc in list_accs]
    for ind, us_text in enumerate(list_us):
        ax[ind][0].text(-10, 120, us_text, rotation=90, horizontalalignment='center', fontsize=18,
                        verticalalignment='center')

    # plt.savefig('/home/studghoul1/lapnet/lapnet/results/LAPNet_cine.png', dpi=1200)

    if save_path:
        plt.savefig(save_path, dpi=1200)
        print(f'evaluation figure is saved under {save_path}')
    if show:
        plt.show()

    return fig


def eval_img(results, show=True, save_path=None):
    fig, ax = plt.subplots(3, 7, figsize=(14, 14))
    plt.axis('off')
    # add images
    for i, data in enumerate(results):
        ax[i][0].imshow(np.abs(data['img_ref']), cmap='gray')
        ax[i][0].set_title('Ref Img')
        ax[i][1].imshow(np.abs(data['img_mov']), cmap='gray')
        ax[i][1].set_title('Moving Img')
        ax[i][2].imshow(np.abs(data['mov_corr']), cmap='gray')
        ax[i][2].set_title('Moving Corrected')

        ax[i][3].imshow(np.abs(data['err_orig']), cmap='gray')
        ax[i][3].set_title('Ref - Moving')

        ax[i][4].imshow(np.abs(data['err_pred']), cmap='gray')
        ax[i][4].set_title('Ref - Pred')

        ax[i][5].imshow(data['color_flow_pred'])
        ax[i][5].set_title('Flow Pred')
        EPE = 'EPE: ' + '{:.4f}'.format(data['loss_pred'])
        EAE = 'EAE: ' + '{:.4f}'.format(data['loss_ang_pred'])

        photo_loss = 'loss: ' + '{:.5f}'.format(np.abs(data['err_pred']).mean())
        ax[i][5].text(10, 310, photo_loss, horizontalalignment='left', fontsize=12, verticalalignment='center')

    ax[0][6].imshow(results[0]['color_flow_gt'])
    ax[0][6].set_title('Flow LAP masked')
    for i in range(7):
        for j in range(3):
            ax[j][i].axis('off')
            # add US_rates:
    list_us = ['fully sampled', 'acc=8x', 'acc=30x']
    for ind, us_text in enumerate(list_us):
        ax[ind][0].text(-20, 120, us_text, rotation=90, horizontalalignment='center', fontsize=12,
                        verticalalignment='center')

    if save_path:
        plt.savefig(f"{save_path}", bbox_inches='tight')
        print(f'evaluation figure is saved under {save_path}')
    if show:
        plt.show()
    return fig
