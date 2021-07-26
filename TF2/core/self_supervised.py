from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras
from core.lapnet import squeeze_func
import tensorflow as tf
import tensorflow_addons as tfa
import neurite as ne


def buildLAPNet_model_2D_unsupervised(crop_size=33):
    input_shape = (crop_size, crop_size, 2)
    moving = Input(shape=input_shape, )
    reference = Input(shape=input_shape, )
    inputs = concatenate([reference, moving])
    # enc
    initializer = VarianceScaling(scale=2.0)
    x = Conv2D(filters=64,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv1")(inputs)
    x = LeakyReLU(alpha=0.1, name='act1')(x)
    x = Conv2D(filters=128,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv2")(x)
    x = LeakyReLU(alpha=0.1, name='act2')(x)
    x = Conv2D(filters=256,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv2_1")(x)

    x = LeakyReLU(alpha=0.1, name='act2_1')(x)
    x = Conv2D(filters=512,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv3_1")(x)
    x = LeakyReLU(alpha=0.1, name='act3_1')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv4")(x)
    x = LeakyReLU(alpha=0.1, name='act4')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="conv4_1")(x)
    x = LeakyReLU(alpha=0.1, name='act4_1')(x)
    x = MaxPooling2D(pool_size=5, name='pool')(x)
    x = Conv2D(2, [1, 1], name="fc2")(x)
    x = Lambda(squeeze_func, name="fc8/squeezed")(x)

    # deform moving image
    flow = Lambda(affine_to_flow, name='affine_to_flow')([moving, x])
    correctedMov = Lambda(Mapping, name='registration')([moving, flow])

    # create the model
    LAPNet_unsupervised = keras.Model(inputs=(reference, moving), outputs=correctedMov)

    return LAPNet_unsupervised


def affine_to_flow(input):
    moving = input[0]
    flow = input[1]
    shape = moving.shape[1:3]
    fun = lambda x: affine_to_dense_shift(x, shape)
    trf = tf.map_fn(fun, flow, dtype=tf.float32)
    return trf


def affine_to_dense_shift(flow, shape, crop_size=33):
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing='ij')
    mesh = [tf.cast(f, 'float32') for f in mesh]
    mesh_ones = tf.ones((1, crop_size, crop_size))
    flow_exp = tf.expand_dims(flow, axis=-1)
    flow_exp = tf.expand_dims(flow_exp, axis=-1)
    loc_matrix = tf.multiply(flow_exp, mesh_ones)
    loc = tf.transpose(loc_matrix, [1, 2, 0])
    res = loc + tf.stack(mesh, axis=-1)
    return res


def Mapping(input):
    a = input[0]
    b = input[1]
    output = tfa.image.dense_image_warp(a, b, name='imag_image_warp')
    return output
