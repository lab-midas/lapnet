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
    flow = Lambda(vector_to_flow, name='affine_to_flow')([moving, x])
    correctedMov = Lambda(Mapping, name='registration')([moving, flow])

    # create the self_supervised model
    LAPNet_self_supervised = keras.Model(inputs=(reference, moving), outputs=correctedMov)

    return LAPNet_self_supervised


# vector_to_flow_func function in layer
def vector_to_flow(input):
    moving = input[0]
    flow = input[1]
    shape = moving.shape[1:3]
    fun = lambda x: vector_to_flow_func(x, shape)
    trf = tf.map_fn(fun, flow, dtype=tf.float32)
    return trf


# create the flow for the entire moving image from the flow vector at the center
def vector_to_flow_func(flow, shape, crop_size=33):
    # define the locations grid
    grid = ne.utils.volshape_to_meshgrid(shape, indexing='ij')
    grid = [tf.cast(f, 'float32') for f in grid]

    # all one matrix to be transformed to the actual flow
    arr_ones = tf.ones((1, crop_size, crop_size))

    # expand the dimension of the vector flow for multiplication
    flow_exp = tf.expand_dims(flow, axis=-1)
    flow_exp = tf.expand_dims(flow_exp, axis=-1)

    # get the extended locations matrix
    ind_arr = tf.multiply(flow_exp, arr_ones)

    # adjust the dimensions to keep the channels at the back
    extended_flow = tf.transpose(ind_arr, [1, 2, 0])

    # get motion field
    res = extended_flow + tf.stack(grid, axis=-1)
    return res


# Warp the moving image with the dense flow to get the corrected patch
def Mapping(input):
    a = input[0]
    b = input[1]
    output = tfa.image.dense_image_warp(a, b, name='imag_image_warp')
    return output
