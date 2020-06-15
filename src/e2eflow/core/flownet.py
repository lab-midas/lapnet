import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


from ..ops import correlation
from .image_warp import image_warp

from .flow_util import flow_to_color


FLOW_SCALE = 5


def flownet(im1, im2, flownet_spec='S', full_resolution=False, train_all=False, LAP_layer=False,
            backward_flow=False):
    num_batch, height, width, _ = tf.unstack(tf.shape(im1))
    flownet_num = len(flownet_spec)
    assert flownet_num > 0
    flows_fw = []
    flows_bw = []
    for i, name in enumerate(flownet_spec):
        assert name in ('C', 'c', 'S', 's')
        channel_mult = 1 if name in ('C', 'S') else 3 / 8
        full_res = full_resolution and i == flownet_num - 1

        def scoped_block():
            if name.lower() == 'c':
                assert i == 0, 'FlowNetS must be used for refinement networks'

                with tf.variable_scope('flownet_c_features'):
                    _, conv2_a, conv3_a = flownet_c_features(im1, channel_mult=channel_mult)
                    _, conv2_b, conv3_b = flownet_c_features(im2, channel_mult=channel_mult, reuse=True)

                with tf.variable_scope('flownet_c') as scope:
                    flow_fw = flownet_c(conv3_a, conv3_b, conv2_a,
                                        full_res=full_res,
                                        channel_mult=channel_mult)
                    flows_fw.append(flow_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw = flownet_c(conv3_b, conv3_a, conv2_b,
                                            full_res=full_res,
                                            channel_mult=channel_mult,
                                            LAP_layer=LAP_layer)
                        flows_bw.append(flow_bw)
            elif name.lower() == 's':
                def _flownet_s(im1, im2, flow=None):
                    if flow is not None:
                        flow = tf.image.resize_bilinear(flow, [height, width]) * 4 * FLOW_SCALE
                        warp = image_warp(im2, flow)
                        diff = tf.abs(warp - im1)
                        if not train_all:
                            flow = tf.stop_gradient(flow)
                            warp = tf.stop_gradient(warp)
                            diff = tf.stop_gradient(diff)

                        inputs = tf.concat([im1, im2, flow, warp, diff], axis=3)
                        inputs = tf.reshape(inputs, [num_batch, height, width, 14])
                    else:
                        inputs = tf.concat([im1, im2], 3)
                    return flownet_s(inputs,
                                     full_res=full_res,
                                     channel_mult=channel_mult,
                                     LAP_layer=LAP_layer)
                stacked = len(flows_fw) > 0
                with tf.variable_scope('flownet_s') as scope:
                    flow_fw = _flownet_s(im1, im2, flows_fw[-1][0] if stacked else None)
                    flows_fw.append(flow_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw = _flownet_s(im2, im1, flows_bw[-1][0] if stacked else None)
                        flows_bw.append(flow_bw)

        if i > 0:
            scope_name = "stack_{}_flownet".format(i)
            with tf.variable_scope(scope_name):
                scoped_block()
        else:
            scoped_block()

    if backward_flow:
        return flows_fw, flows_bw
    return flows_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)


def _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, channels=2):
    m = channel_mult

    flow6 = slim.conv2d(conv6_1, channels, 3, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, flow6_up5], 1)
    flow5 = slim.conv2d(concat5, channels, 3, scope='flow5',
                       activation_fn=None)

    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, flow5_up4], 1)
    flow4 = slim.conv2d(concat4, channels, 3, scope='flow4',
                       activation_fn=None)

    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, flow4_up3], 1)
    flow3 = slim.conv2d(concat3, channels, 3, scope='flow3',
                       activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, flow3_up2], 1)

    flow2 = slim.conv2d(concat2, channels, 3, scope='flow2',
                       activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, flow2_up1], 1)
            flow1 = slim.conv2d(concat1, channels, 3, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, flow1_up0], 1)
            flow0 = slim.conv2d(concat0, channels, 3, scope='flow0',
                                activation_fn=None)

            flows = [flow0, flow1] + flows

    return flows


def nhwc_to_nchw(tensors):
    return [tf.transpose(t, [0, 3, 1, 2]) for t in tensors]


def nchw_to_nhwc(tensors):
    return [tf.transpose(t, [0, 2, 3, 1]) for t in tensors]


def flownet_s_kspace(inputs, channel_mult=1, full_res=False, LAP_layer=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        if LAP_layer:
            channels = 3
        else:
            channels = 2
        conv6_1_i = tf.complex(conv6_1[:, :tf.cast(tf.shape(conv6_1)[1]/2, dtype=tf.int32), :, :],
                               conv6_1[:, tf.cast(tf.shape(conv6_1)[1]/2, dtype=tf.int32):, :, :])
        conv5_1_i = tf.complex(conv5_1[:, :tf.cast(tf.shape(conv5_1)[1] / 2, dtype=tf.int32), :, :],
                               conv5_1[:, tf.cast(tf.shape(conv5_1)[1] / 2, dtype=tf.int32):, :, :])
        conv4_1_i = tf.complex(conv4_1[:, :tf.cast(tf.shape(conv4_1)[1] / 2, dtype=tf.int32), :, :],
                               conv4_1[:, tf.cast(tf.shape(conv4_1)[1] / 2, dtype=tf.int32):, :, :])
        conv3_1_i = tf.complex(conv3_1[:, :tf.cast(tf.shape(conv3_1)[1] / 2, dtype=tf.int32), :, :],
                               conv3_1[:, tf.cast(tf.shape(conv3_1)[1] / 2, dtype=tf.int32):, :, :])
        conv2_i = tf.complex(conv2[:, :tf.cast(tf.shape(conv2)[1] / 2, dtype=tf.int32), :, :],
                             conv2[:, tf.cast(tf.shape(conv2)[1] / 2, dtype=tf.int32):, :, :])

        conv6_1_r = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(conv6_1_i, axes=(-2, -1))))
        conv5_1_r = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(conv5_1_i, axes=(-2, -1))))
        conv4_1_r = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(conv4_1_i, axes=(-2, -1))))
        conv3_1_r = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(conv3_1_i, axes=(-2, -1))))
        conv2_r = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(conv2_i, axes=(-2, -1))))
        res = _flownet_upconv(conv6_1_r, conv5_1_r, conv4_1_r, conv3_1_r, conv2_r, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res, channels=channels)
        return nchw_to_nhwc(res)


def flownet_s(inputs, channel_mult=1, full_res=False, LAP_layer=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    m = 1  # todo
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        if LAP_layer:
            channels = 3
        else:
            channels = 2
        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res, channels=channels)
        return nchw_to_nhwc(res)


def flownet_s_kspace_in_33_out_4(inputs, channel_mult=1, full_res=False):

    m = channel_mult
    # m = 3 / 8
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv2_1 = slim.conv2d(conv2, int(256 * m), 5, stride=1, scope='conv2_1')
        # conv3 = slim.conv2d(conv2_1, int(512 * m), 3, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv2_1, int(512 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(1024 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(1024 * m), 3, stride=1, scope='conv4_1')

    pool = slim.max_pool2d(conv4_1, 5, data_format='NCHW')
    flatten_conv6_1 = slim.flatten(pool)

    # fc1 = slim.fully_connected(flatten_conv6_1,
    #                            4096,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.constant_initializer(0.1),
    #                            scope='fc1')
    # dp1 = slim.dropout(fc1, 0.5, is_training=True,
    #                    scope='dropout6')

    # fc2 = slim.fully_connected(flatten_conv6_1,
    #                            2,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.zeros_initializer(),
    #                            activation_fn=None,
    #                            scope='fc2')
    fc2 = slim.conv2d(pool,
                      4, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.compat.v1.zeros_initializer(),
                      data_format='NCHW',
                      scope='fc2')
    fc2 = tf.squeeze(fc2, name='fc8/squeezed', axis=[-1, -2])

    return fc2


def flownet_s_kspace_in_33(inputs, channel_mult=1, full_res=False):
    m = channel_mult
    # m = 3 / 8
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv2_1 = slim.conv2d(conv2, int(256 * m), 5, stride=1, scope='conv2_1')
        # conv3 = slim.conv2d(conv2_1, int(512 * m), 5, stride=1, scope='conv3')
        conv3_1 = slim.conv2d(conv2_1, int(512 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(1024 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(1024 * m), 3, stride=1, scope='conv4_1')

    pool = slim.max_pool2d(conv4_1, 5, data_format='NCHW')
    # flatten_conv6_1 = slim.flatten(pool)

    fc2 = slim.conv2d(pool,
                      2, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.compat.v1.zeros_initializer(),
                      data_format='NCHW',
                      scope='fc2')
    fc2 = tf.squeeze(fc2, name='fc8/squeezed', axis=[-1, -2])

    return fc2


def flownet_s_kspace_in_33_large_receptive(inputs, channel_mult=1, full_res=False):

    m = channel_mult
    # m = 3 / 8
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 17, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 9, stride=1, scope='conv2')
        conv2_1 = slim.conv2d(conv2, int(256 * m), 9, stride=2, scope='conv2_1')
        conv3 = slim.conv2d(conv2_1, int(512 * m), 5, stride=1, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(512 * m), 5, stride=2, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(1024 * m), 3, stride=1, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(1024 * m), 3, stride=1, scope='conv4_1')

    pool = slim.max_pool2d(conv4_1, 5, data_format='NCHW')
    flatten_conv6_1 = slim.flatten(pool)

    # fc1 = slim.fully_connected(flatten_conv6_1,
    #                            4096,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.constant_initializer(0.1),
    #                            scope='fc1')
    # dp1 = slim.dropout(fc1, 0.5, is_training=True,
    #                    scope='dropout6')

    # fc2 = slim.fully_connected(flatten_conv6_1,
    #                            2,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.zeros_initializer(),
    #                            activation_fn=None,
    #                            scope='fc2')
    fc2 = slim.conv2d(pool,
                      2, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.compat.v1.zeros_initializer(),
                      data_format='NCHW',
                      scope='fc2')
    fc2 = tf.squeeze(fc2, name='fc8/squeezed', axis=[-1, -2])

    return fc2


def flownet_s_kspace_in_65(inputs, channel_mult=1, full_res=False):

    m = channel_mult
    # m = 3 / 8
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv2_1 = slim.conv2d(conv2, int(256 * m), 5, stride=1, scope='conv2_1')
        conv3 = slim.conv2d(conv2_1, int(512 * m), 3, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(512 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(1024 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(1024 * m), 3, stride=1, scope='conv4_1')

    pool = slim.max_pool2d(conv4_1, 5, data_format='NCHW')
    flatten_conv6_1 = slim.flatten(pool)

    # fc1 = slim.fully_connected(flatten_conv6_1,
    #                            4096,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.constant_initializer(0.1),
    #                            scope='fc1')
    # dp1 = slim.dropout(fc1, 0.5, is_training=True,
    #                    scope='dropout6')

    # fc2 = slim.fully_connected(flatten_conv6_1,
    #                            2,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.zeros_initializer(),
    #                            activation_fn=None,
    #                            scope='fc2')
    fc2 = slim.conv2d(pool,
                      2, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.compat.v1.zeros_initializer(),
                      data_format='NCHW',
                      scope='fc2')
    fc2 = tf.squeeze(fc2, name='fc8/squeezed', axis=[-1, -2])

    return fc2


def flownet_s_kspace_in_full(inputs, channel_mult=1, full_res=False):

    m = channel_mult
    # m = 3 / 8
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')


    pool = slim.max_pool2d(conv6_1, 4, data_format='NCHW')
    flatten_conv6_1 = slim.flatten(pool)

    # fc1 = slim.fully_connected(flatten_conv6_1,
    #                            4096,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.constant_initializer(0.1),
    #                            scope='fc1')
    # dp1 = slim.dropout(fc1, 0.5, is_training=True,
    #                    scope='dropout6')

    # fc2 = slim.fully_connected(flatten_conv6_1,
    #                            2,
    #                            weights_initializer=layers.variance_scaling_initializer(),
    #                            biases_initializer=tf.zeros_initializer(),
    #                            activation_fn=None,
    #                            scope='fc2')
    fc2 = slim.conv2d(pool,
                      2, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.compat.v1.zeros_initializer(),
                      data_format='NCHW',
                      scope='fc2')
    fc2 = tf.squeeze(fc2, name='fc8/squeezed', axis=[-1, -2])

    return fc2


def flownet_c_features(im, channel_mult=1, reuse=None):
    m = channel_mult
    im = nhwc_to_nchw([im])[0]
    with slim.arg_scope([slim.conv2d],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(im, int(64 * m), 7, stride=2, scope='conv1', reuse=reuse)
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2', reuse=reuse)
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3', reuse=reuse)
        return conv1, conv2, conv3


def flownet_c(conv3_a, conv3_b, conv2_a, channel_mult=1, full_res=False, LAP_layer=False):
    """Given two images, returns flow predictions in decreasing resolution.

    Uses FlowNetCorr.
    """
    m = channel_mult

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        corr = correlation(conv3_a, conv3_b,
                           pad=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2)

        conv_redir = slim.conv2d(conv3_a, int(32 * m), 1, stride=1, scope='conv_redir')

        conv3_1 = slim.conv2d(tf.concat([conv_redir, corr], 1), int(256 * m), 3,
                              stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        if LAP_layer:
            channels = 3
        else:
            channels = 2

        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2_a,
                              channel_mult=channel_mult, full_res=full_res, channels=channels)
        return nchw_to_nhwc(res)
