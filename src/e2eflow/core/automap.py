import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import numpy as np


def automap(x):
    """ Defines all layers for forward propagation:
    Fully connected (FC1) -> tanh activation: size (n_im, n_H0 * n_W0)
    -> Fully connected (FC2) -> tanh activation:  size (n_im, n_H0 * n_W0)
    -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
    -> Convolutional -> ReLU activation with l1 regularization: size (n_im, n_H0, n_W0, 64)
    -> De-convolutional: size (n_im, n_H0, n_W0)
    :param x: Input - images in frequency space, size (n_im, n_H0, n_W0, 2)
    :return: output of the last layer of the neural network
    """

    x_temp = tf.contrib.layers.flatten(x)  # size (n_im, n_H0 * n_W0 * 4)
    n_out = np.int(x.shape[1] * x.shape[2] * x.shape[3]) // 8  # size (n_im, n_H0 * n_W0 * 2)

    # FC: input size (n_im, n_H0 * n_W0 * 2), output size (n_im, n_H0 * n_W0)
    FC1 = layers.fully_connected(
        x_temp,
        n_out * 2,
        activation_fn=_leaky_relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(0.0004),
        biases_initializer=None,
        biases_regularizer=None,
        reuse=tf.AUTO_REUSE,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc1')

    # FC: input size (n_im, n_H0 * n_W0), output size (n_im, n_H0 * n_W0)
    FC2 = layers.fully_connected(
        FC1,
        n_out * 2,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(0.0004),
        biases_initializer=None,
        biases_regularizer=None,
        reuse=tf.AUTO_REUSE,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc2')

    # FC: input size (n_im, n_H0 * n_W0), output size (n_im, n_H0 * n_W0)
    FC3 = layers.fully_connected(
        FC2,
        n_out,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=None,
        biases_regularizer=None,
        reuse=tf.AUTO_REUSE,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc3')

    # FC4 = layers.fully_connected(
    #     FC3,
    #     n_out,
    #     activation_fn=tf.tanh,
    #     normalizer_fn=None,
    #     normalizer_params=None,
    #     weights_initializer=tf.contrib.layers.xavier_initializer(),
    #     weights_regularizer=None,
    #     biases_initializer=None,
    #     biases_regularizer=None,
    #     reuse=tf.AUTO_REUSE,
    #     variables_collections=None,
    #     outputs_collections=None,
    #     trainable=True,
    #     scope='fc4')
    #
    # FC5 = layers.fully_connected(
    #     FC4,
    #     n_out,
    #     activation_fn=tf.tanh,
    #     normalizer_fn=None,
    #     normalizer_params=None,
    #     weights_initializer=tf.contrib.layers.xavier_initializer(),
    #     weights_regularizer=None,
    #     biases_initializer=None,
    #     biases_regularizer=None,
    #     reuse=tf.AUTO_REUSE,
    #     variables_collections=None,
    #     outputs_collections=None,
    #     trainable=True,
    #     scope='fc5')

    # Reshape output from FC layers into array of size (n_im, n_H0, n_W0, 1):
    FC_M = tf.reshape(FC3, [tf.shape(x)[0], tf.shape(x)[1] // 2, tf.shape(x)[2] // 2, 2])



    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        # Input size (n_im, n_H0, n_W0, 1), output size (n_im, n_H0, n_W0, 64)
        conv1 = slim.conv2d(FC_M, 2, 3, stride=1, padding='SAME', activation_fn=None, scope='conv1')

        # Input size (n_im, n_H0, n_W0, 64), output size (n_im, n_H0, n_W0, 64)
        #conv2 = slim.conv2d(conv1, 2, 5, stride=1, padding='SAME', activation_fn=_leaky_relu, scope='conv2')
        #conv3 = slim.conv2d(conv2, )
    #
    # CONV3 = slim.conv2d(
    #     conv2,
    #     filters=1,
    #     kernel_size=7,
    #     strides=1,
    #     padding='same',
    #     data_format='channels_last',
    #     dilation_rate=(1, 1),
    #     activation=tf.nn.relu,
    #     use_bias=True,
    #     kernel_initializer=None,
    #     bias_initializer=tf.zeros_initializer(),
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     #        activity_regularizer=tf.contrib.layers.l1_regularizer(0.0001),
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     trainable=True,
    #     name='conv3',
    #     reuse=tf.AUTO_REUSE)
    #
    # DECONV = tf.squeeze(CONV3)

    return conv1

def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)
