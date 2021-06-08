import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .augment import random_photometric
from .flow_util import flow_to_color
from .losses import charbonnier_loss
from .flownet import flownet, flownet_s_kspace_in_full, flownet_s_kspace_in_65, flownet_s_kspace_in_33
from .automap import automap
from .unsupervised import _track_image, _track_loss, FLOW_SCALE


def _track_param(op, name):
    tf.add_to_collection('params', tf.identity(op, name=name))


def supervised_loss(batch, params, augment=False):
    im1, im2, flow_gt = batch

    im1 = im1[:, 0, :, :]
    im2 = im2[:, 0, :, :]
    if len(flow_gt.get_shape()) is 5:
        flow_gt = flow_gt[:, 0, :, :, :]
    elif len(flow_gt.get_shape()) is 3:
        flow_gt = flow_gt[:, 0, :]

    if len(im1.get_shape()) is 3:
        im1 = im1[..., tf.newaxis]
        im2 = im2[..., tf.newaxis]

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

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res')
    train_all = params.get('train_all')
    # -------------------------------------------------------------------------

    if params.get('network') == 'flownet':
        # FlowNet
        flows_fw = flownet(im1_photo, im2_photo,
                           flownet_spec=flownet_spec,
                           full_resolution=full_resolution,
                           train_all=train_all,
                           LAP_layer=params.get('lap_layer'))

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
            # _track_image(flow_to_color(final_flow_fw), 'flow_pred_' + str(i))

            flow_x, _ = tf.split(axis=3, num_or_size_splits=2, value=final_flow_fw)
            mean_flow_x = tf.reduce_mean(flow_x)
            flow_x_gt, _ = tf.split(axis=3, num_or_size_splits=2, value=flow_gt)
            mean_flow_x_gt = tf.reduce_mean(flow_x_gt)
            _track_loss(mean_flow_x, 'mean_flow_x')
            _track_loss(mean_flow_x_gt, 'mean_flow_x_gt')

            net_loss = charbonnier_loss(final_flow_fw - flow_gt, mask=None)
            # layer_weight = layer_weights[i]
            final_loss += net_loss / (2 ** i)

        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        final_loss += regularization_loss
        _track_loss(regularization_loss, 'loss/regularization')
        _track_loss(final_loss, 'loss/combined')
    elif params.get('network') == 'lapnet':
        inputs = tf.concat([im1_photo, im2_photo], 3)
        if not params.get('automap'):
            with tf.variable_scope('flownet_s'):
                if im1.get_shape().as_list()[2] is 256:
                    final_flow_fw = flownet_s_kspace_in_full(inputs, channel_mult=1)
                elif im1.get_shape().as_list()[2] is 65:
                    final_flow_fw = flownet_s_kspace_in_65(inputs, channel_mult=1)
                elif im1.get_shape().as_list()[2] is 33:
                    final_flow_fw = flownet_s_kspace_in_33(inputs, channel_mult=1)

                if len(flow_gt.get_shape()) is 4:
                    flow_gt = flow_gt[:, 0, 0, :]

                error = final_flow_fw - flow_gt
                # final_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(error), 1)))
                final_loss = tf.reduce_mean(tf.square(error))

                # final_loss = charbonnier_loss(error, mask=None)
                regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
                final_loss += regularization_loss

                _track_loss(final_loss, 'loss/combined')
                flow_x, _ = tf.split(axis=1, num_or_size_splits=2, value=final_flow_fw)
                mean_flow_x = tf.reduce_mean(flow_x)
                flow_x_gt, _ = tf.split(axis=1, num_or_size_splits=2, value=flow_gt)
                mean_flow_x_gt = tf.reduce_mean(flow_x_gt)
                _track_loss(mean_flow_x, 'mean_flow_x')
                _track_loss(mean_flow_x_gt, 'mean_flow_x_gt')
        else:
            with tf.variable_scope('flownet_s'):
                final_flow_fw = automap(inputs)
                final_flow_fw = tf.image.resize_bilinear(final_flow_fw, im_shape) * FLOW_SCALE * 2
                flow_x, _ = tf.split(axis=3, num_or_size_splits=2, value=final_flow_fw)
                mean_flow_x = tf.reduce_mean(flow_x)
                flow_x_gt, _ = tf.split(axis=3, num_or_size_splits=2, value=flow_gt)
                mean_flow_x_gt = tf.reduce_mean(flow_x_gt)
                _track_loss(mean_flow_x, 'mean_flow_x')
                _track_loss(mean_flow_x_gt, 'mean_flow_x_gt')
                flow_gt = tf.cast(flow_gt, dtype=tf.float32)

                error = final_flow_fw - flow_gt
                # final_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(error), 1)))
                final_loss = tf.reduce_mean(tf.square(error))

                regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
                final_loss += regularization_loss
                _track_loss(regularization_loss, 'loss/regularization')
                _track_loss(final_loss, 'loss/combined')

    return final_loss, final_flow_fw


