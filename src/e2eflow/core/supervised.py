import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .augment import random_photometric
from .flow_util import flow_to_color
from .losses import charbonnier_loss
from .flownet import flownet
from .unsupervised import _track_image, _track_loss, FLOW_SCALE


def supervised_loss(batch, params, normalization=None, LAP=False, augment=False):
    # channel_mean = tf.constant(normalization[0]) / 255.0
    im1, im2, flow_gt = batch
    # im1, im2, flow_gt, mask_gt = tf.squeeze(im1), tf.squeeze(im2), tf.squeeze(flow_gt), tf.squeeze(mask_gt)

    im1 = im1[:, 0, :, :]
    im2 = im2[:, 0, :, :]
    flow_gt = flow_gt[:, 0, :, :, :]
    # mask_gt = mask_gt[:, 0, :, :]

    im1 = im1[..., tf.newaxis]
    im2 = im2[..., tf.newaxis]
    # mask_gt = mask_gt[..., tf.newaxis]

    # im1 = batch[..., 0]
    # im2 = batch[..., 1]
    # flow_gt = batch[..., 2:4]
    # mask_gt = batch[..., 4]
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]

    # -------------------------------------------------------------------------
    if augment:
        im1_photo, im2_photo = random_photometric(
            [im1, im2],
            noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
            brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
            min_gamma=0.7, max_gamma=1.5)
        _track_image(im1_photo, 'im1_photo')
        _track_image(im2_photo, 'im2_photo')
        _track_image(flow_to_color(flow_gt), 'flow_gt')
        # _track_image(mask_gt, 'mask_gt')
    else:
        im1_photo, im2_photo = im1, im2



    # Images for neural network input with mean-zero values in [-1, 1]
    # im1_photo = im1_photo - channel_mean
    # im2_photo = im2_photo - channel_mean

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res')
    train_all = params.get('train_all')
    # -------------------------------------------------------------------------
    # FlowNet
    flows_fw = flownet(im1_photo, im2_photo,
                       flownet_spec=flownet_spec,
                       full_resolution=full_resolution,
                       train_all=train_all)
    
    if not train_all:
        flows_fw = [flows_fw[-1]]
    final_loss = 0.0
    # layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_weights = [1, 1, 1, 1, 1]
    for i, net_flows in enumerate(reversed(flows_fw)):
        flow_fw = net_flows[0]
        if params.get('full_res'):
            final_flow_fw = flow_fw * FLOW_SCALE * 4
        else:
            final_flow_fw = tf.image.resize_bilinear(flow_fw, im_shape) * FLOW_SCALE * 4
        _track_image(flow_to_color(final_flow_fw), 'flow_pred_' + str(i))

        if LAP:
            final_flow_fw = slim.conv2d(final_flow_fw, 1, 4, stride=1)
            # final_flow_fw = tf.reduce_sum(final_flow_fw, axis=3)  # todo: find a better way!
            # final_flow_fw = tf.expand_dims(final_flow_fw, -1)
            paddings = tf.constant([[0, 0, ], [1, 1, ], [1, 1], [0, 0, ]])
            final_flow_fw = tf.pad(final_flow_fw, paddings, 'SYMMETRIC')
            filter_x = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])
            filter_y = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
            filter_x = np.expand_dims(filter_x, -1)
            filter_x = np.expand_dims(filter_x, -1)
            filter_y = np.expand_dims(filter_y, -1)
            filter_y = np.expand_dims(filter_y, -1)
            a = tf.nn.conv2d(final_flow_fw, filter_x, strides=[1, 1, 1, 1], padding='VALID')
            b = tf.nn.conv2d(final_flow_fw, filter_y, strides=[1, 1, 1, 1], padding='VALID')
            flow_x = a / tf.reduce_sum(final_flow_fw)
            flow_y = b / tf.reduce_sum(final_flow_fw)

            final_flow_fw = tf.concat([flow_x, flow_y], axis=3)

        net_loss = charbonnier_loss(final_flow_fw - flow_gt, mask=None)
        # layer_weight = layer_weights[i]
        final_loss += net_loss / (2 ** i)

    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
    final_loss += regularization_loss
    _track_loss(regularization_loss, 'loss/regularization')
    _track_loss(final_loss, 'loss/combined')

    return final_loss, final_flow_fw
