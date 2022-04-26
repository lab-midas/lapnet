import os
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Lambda, Input, AveragePooling2D, Layer
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras
import sys
sys.path.append('/home/studghoul1/lapnet/lapnet/core')
from tapering import reshuffle, fft_along_dim, ifft_along_dim
from scipy.interpolate import interpn
import numpy as np
#from optotf.keras.warp import Warp, WarpTranspose


def buildLAPNet_model_2D(crop_size=33):
    input_shape = (crop_size, crop_size, 4)
    inputs = Input(shape=input_shape, )
    model = keras.Sequential()
    initializer = VarianceScaling(scale=2.0)
    model.add(inputs)

    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv1"))
    model.add(LeakyReLU(alpha=0.1, name='act1'))
    model.add(Conv2D(filters=128,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2"))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Conv2D(filters=256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2_1"))
    model.add(LeakyReLU(alpha=0.1, name='act2_1'))
    model.add(Conv2D(filters=512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv3_1"))
    model.add(LeakyReLU(alpha=0.1, name='act3_1'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4"))
    model.add(LeakyReLU(alpha=0.1, name='act4'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4_1"))

    model.add(LeakyReLU(alpha=0.1, name='act4_1'))
    model.add(MaxPooling2D(pool_size=5, name='pool'))
    model.add(Conv2D(2, 1, name="fc2"))
    model.add(Lambda(squeeze_func, name="fc8/squeezed"))
    return model


def buildLAPNET_model_2D_with_tapering(rows=256, cols=256):
    input_shape = (rows, cols)
    initializer = VarianceScaling(scale=2.0)
    ref_input = tf.keras.Input(shape=input_shape, name='ref_input', dtype=tf.complex128)
    mov_input = tf.keras.Input(shape=input_shape, name='mov_input', dtype=tf.complex128)
    x = TaperingLayer()([ref_input, mov_input])
    x = Lambda(lambda x: tf.concat((x[0], x[1]), axis=-1), name='stack_patches')(x)
    x = Conv2D(filters=64,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv1")(x)
    x = LeakyReLU(alpha=0.1, name='c_act1')(x)
    x = Conv2D(filters=128,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv2")(x)
    x = LeakyReLU(alpha=0.1, name='c_act2')(x)
    x = Conv2D(filters=256,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv2_1")(x)
    x = Conv2D(filters=512,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv3_1")(x)
    x = LeakyReLU(alpha=0.1, name='c_act3_1')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv4")(x)
    x = LeakyReLU(alpha=0.1, name='c_act4')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(0.0004),
               kernel_initializer=initializer,
               name="c_conv4_1")(x)
    x = LeakyReLU(alpha=0.1, name='c_act4_1')(x)
    x = AveragePooling2D(pool_size=5, name='pool_c')(x)
    x = Conv2D(2, [1, 1], name="c_fc3")(x)
    x = Lambda(squeeze_func, name="fc8/squeezed_c")(x)

    #
    model = tf.keras.Model(inputs=[ref_input, mov_input], outputs=x)
    return model


# Model with descendent kernel sizes
def buildLAPNet_model_2D_old(crop_size=33):
    initializer = VarianceScaling(scale=2.0)
    input_shape = (crop_size, crop_size, 4)

    inputs = Input(shape=input_shape, )
    model = keras.Sequential()
    model.add(inputs)
    model.add(Conv2D(filters=64,
                     kernel_size=7,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv1"))
    # model.add(Activation(leaky_relu, name='act1'))
    model.add(LeakyReLU(alpha=0.1, name='act1'))
    model.add(Conv2D(filters=128,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2"))
    # model.add(Activation(leaky_relu, name='act2'))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Conv2D(filters=256,
                     kernel_size=5,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2_1"))
    # model.add(Activation(leaky_relu, name='act2_1'))
    model.add(LeakyReLU(alpha=0.1, name='act2_1'))
    model.add(Conv2D(filters=512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv3_1"))
    # model.add(Activation(leaky_relu, name='act3_1'))
    model.add(LeakyReLU(alpha=0.1, name='act3_1'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4"))
    # model.add(Activation(leaky_relu, name='act4'))
    model.add(LeakyReLU(alpha=0.1, name='act4'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4_1"))
    # model.add(Activation(leaky_relu, name='act4_1'))
    model.add(LeakyReLU(alpha=0.1, name='act4_1'))
    model.add(MaxPooling2D(pool_size=5, name='pool'))
    model.add(Conv2D(2, [1, 1], name="fc2"))
    model.add(Lambda(squeeze_func, name="fc8/squeezed"))
    return model


# design the 3D model
def buildLAPNet_model_3D():
    initializer = VarianceScaling(scale=2.0)

    input_shape = (33, 33, 4)
    coronal_input = Input(shape=input_shape, name="coronal")
    sagital_input = Input(shape=input_shape, name="sagital")

    coronal_features = Conv2D(filters=64,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv1")(coronal_input)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act1')(coronal_features)
    coronal_features = Conv2D(filters=128,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv2")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act2')(coronal_features)
    coronal_features = Conv2D(filters=256,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv2_1")(coronal_features)
    coronal_features = Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv3_1")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act3_1')(coronal_features)
    coronal_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv4")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act4')(coronal_features)
    coronal_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv4_1")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act4_1')(coronal_features)
    coronal_features = AveragePooling2D(pool_size=5, name='pool_c')(coronal_features)
    coronal_features = Conv2D(2, [1, 1], name="c_fc3")(coronal_features)

    sagital_features = Conv2D(filters=64,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv1")(sagital_input)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act1')(sagital_features)
    sagital_features = Conv2D(filters=128,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv2")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act2')(sagital_features)
    sagital_features = Conv2D(filters=256,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv2_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act2_1')(sagital_features)
    sagital_features = Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv3_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act3_1')(sagital_features)
    sagital_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv4")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act4')(sagital_features)
    sagital_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv4_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act4_1')(sagital_features)
    sagital_features = AveragePooling2D(pool_size=5, name='pool_s')(sagital_features)
    sagital_features = Conv2D(2, [1, 1], name="s_fc3")(sagital_features)

    final_flow_c = Lambda(squeeze_func, name="fc8/squeezed_c")(coronal_features)
    final_flow_s = Lambda(squeeze_func, name="fc8/squeezed_s")(sagital_features)

    final_flow = WeightedSum(name='final_flow_weighted')([final_flow_c, final_flow_s])
    final_flow = Lambda(squeeze_func, name="squeezed_flow")(final_flow)

    model = keras.Model(
        inputs=[coronal_input, sagital_input],
        outputs=[final_flow],
    )

    return model


# Removes dimensions of size 1 from the shape of the input tensor
def squeeze_func(x):
    try:
        return tf.squeeze(x, axis=[1, 2])
    except:
        try:
            return tf.squeeze(x, axis=[1])
        except Exception as e:
            print(e)


# load checkpoints of trained models in tensorflow 1
def load_tf1_LAPNet_cropping_ckpt(
        checkpoint_path_old='/mnt/data/projects/MoCo/LAPNet/UnFlow/log/ex/resp/srx424_drUS_1603/model.ckpt'):
    model = buildLAPNet_model_2D_old()
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path_old)
    layers_name = ['conv1', 'conv2', 'conv2_1', 'conv3_1', 'conv4', 'conv4_1', 'fc2']
    for i in range(len(layers_name)):
        weights_key = 'flownet_s/' + layers_name[i] + '/weights'
        bias_key = 'flownet_s/' + layers_name[i] + '/biases'
        weights = reader.get_tensor(weights_key)
        biases = reader.get_tensor(bias_key)
        model.get_layer(layers_name[i]).set_weights([weights, biases])  # name the layers
    return model


class WeightedSum(Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(name='weighted',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(0.5),
                                 dtype='float32',
                                 trainable=True,
                                 constraint=tf.keras.constraints.min_max_norm(max_value=1, min_value=0))
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        """return tf.stack((self.a * tf.gather(model_outputs[0], [0], axis=1) + (1 - self.a) * tf.gather(model_outputs[1],
                                                                                                      [0], axis=1),
                         tf.gather(model_outputs[0], [1], axis=1),
                         tf.gather(model_outputs[1], [1], axis=1)), axis=-1)"""
        return tf.stack((tf.gather(model_outputs[0], [0], axis=1) + self.a * tf.gather(model_outputs[1], [0], axis=1),
                         tf.gather(model_outputs[0], [1], axis=1),
                         tf.gather(model_outputs[1], [1], axis=1)), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class getMovLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        _,_,x_real, x_imag = tf.split(x, num_or_size_splits=4, axis=-1)
        try:
            x = tf.complex(tf.squeeze(x_real, -1), tf.squeeze(x_imag, -1))
        except:
            pass
        return x

class ifftLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ifft = Lambda(self.ifftop)

    @tf.function
    def tf_ifftnshift_function(self, input):
        y = tf.numpy_function(ifft_along_dim, [input], tf.complex128)
        return y

    @tf.function
    def ifftop(self, inputs):
        x = tf.map_fn(self.tf_ifftnshift_function, inputs, fn_output_signature=tf.complex128)
        return x

    def call(self, x):
        M, N = x.get_shape().as_list()[-2:]
        x = self.ifft(x)
        x = tf.reshape(x, (-1, 1, M, N))
        # x = tf.stack((tf.math.real(x), tf.math.imag(x)), axis=1)
        return tf.math.real(x), tf.math.imag(x)

def buildLAPNet_model_2D_unsupervised(rows=256, cols=256, patch_size=33, preprocessing=False):
    initializer = VarianceScaling(scale=2.0)
    l2_weight = 0.00001
    if preprocessing:
        input_shape = (rows, cols)
        preprocessingLayer = TaperingLayer()
        ref_input = tf.keras.Input(shape=input_shape, name='ref_input')
        mov_input = tf.keras.Input(shape=input_shape, name='mov_input')
        ref_mov = preprocessingLayer([ref_input, mov_input])
        input = Lambda(lambda x: tf.concat((x[0], x[1]), axis=-1), name='stack_patches')(ref_mov)
    else:
        input_shape = (33, 33, 4)
        input = tf.keras.Input(shape=input_shape, name='k_input')
        mov_img_complex = tf.keras.Input(shape=(2, 33,33))

    x = Conv2D(filters=64,
               kernel_size=7,
               strides=2,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv1")(input)
    x = LeakyReLU(alpha=0.1, name='act1')(x)
    x = Conv2D(filters=128,
               kernel_size=5,
               strides=2,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv2")(x)
    x = LeakyReLU(alpha=0.1, name='act2')(x)
    x = Conv2D(filters=256,
               kernel_size=5,
               strides=1,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv2_1")(x)

    x = LeakyReLU(alpha=0.1, name='act2_1')(x)
    x = Conv2D(filters=512,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv3_1")(x)
    x = LeakyReLU(alpha=0.1, name='act3_1')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv4")(x)
    x = LeakyReLU(alpha=0.1, name='act4')(x)
    x = Conv2D(filters=1024,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_regularizer=l2(l2_weight),
               kernel_initializer=initializer,
               name="conv4_1")(x)
    x = LeakyReLU(alpha=0.1, name='act4_1')(x)

    x = MaxPooling2D(pool_size=5, name='pool')(x)

    x = Conv2D(filters=2,
               kernel_size=1,
               kernel_initializer=initializer,
               kernel_regularizer=l2(l2_weight),
               name="fc2")(x)

    x = tf.keras.layers.UpSampling2D(size=(patch_size, patch_size), name="upsampling")(x)

    x = WrapLayer()(mov_img_complex, x)

    LAPNet_self_supervised = keras.Model(inputs=[input,mov_img_complex], outputs=x)

    return LAPNet_self_supervised


@tf.custom_gradient
def custom_warp(image, flow):
    flow = tf.stack((flow, flow), 1)
    M, N = image.get_shape().as_list()[-3:-1]
    image = tf.reshape(image, (-1, 1, M, N))
    flow = tf.reshape(flow, (-1, M, N, 2))
    W = Warp(channel_last=False)
    Wx = W(image, flow)
    out = tf.reshape(Wx, (-1, M, N, 2))

    def custom_grad(dx):
        dx = tf.reshape(dx, (-1, 1, M, N))
        u = tf.reshape(flow, (-1, M, N, 2))
        x_warpT = WarpTranspose(channel_last=False)(dx, u)
        x_warpT = tf.reshape(x_warpT, (-1, M, N, 2))
        return [None, x_warpT]

    return out, custom_grad


class WrapLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def call(self, image, flow):
        x = custom_warp(image, flow)
        # x = tf.squeeze(x, [2])
        return x

class TaperingLayer(tf.keras.layers.Layer):
  def __init__(self, patch_size=33, rows=256, cols=256, pad=True):
    super(TaperingLayer, self).__init__()
    self.rows, self.cols = rows, cols
    self.patch_size = patch_size
    self.pad = pad
    if self.pad:
        self.pad_Value = self.rows - self.patch_size
        self.windim = 2 * self.rows - self.patch_size
        self.win = self.create_win()
    else:
        self.pad_Value = int((self.rows - self.patch_size)/2)
        self.windim = self.rows

    self.gridxy, self.gridxyq = self.calc_grid_values()
    self.fft_layer = Lambda(self.fftop)
    self.conv_layer =  Lambda(self.convop)
    self.tapering_layer = Lambda(self.taperingop)
    self.regrid_layer = Lambda(self.regridop)
    self.stack_layer = Lambda(self.stackop)

  def create_win_np(self, x_pos, y_pos):
        win = np.zeros((self.rows, self.cols), dtype=np.complex128)
        win[x_pos:x_pos + self.patch_size, y_pos:y_pos + self.patch_size] = 1
        return win

  @tf.function
  def create_custom_win(self):
      x_pos = np.random.randint(0, self.rows - self.patch_size + 1)
      y_pos = np.random.randint(0, self.cols - self.patch_size + 1)
      win = tf.numpy_function(self.create_win_np, [x_pos, y_pos], tf.complex128)
      win = self.tf_fftnshift_function(win)
      return win

  def create_win(self):
      win = tf.ones((self.patch_size, self.patch_size), dtype=np.float32)
      paddings = [[self.pad_Value, self.pad_Value], [self.pad_Value, self.pad_Value]]
      win = tf.pad(win, paddings, "CONSTANT")
      win = self.tf_fftnshift_function(win)
      return win

  def calc_grid_values(self):
      grid_x, grid_y = np.mgrid[0:self.windim], np.mgrid[0:self.windim]
      x = np.linspace(0, self.windim - 1, num=self.patch_size)
      y = np.linspace(0, self.windim - 1, num=self.patch_size)
      grid_xq, grid_yq = np.meshgrid(x, y)
      return (grid_x, grid_y), (grid_yq, grid_xq)

  def fftnshift(self, x):
      output =  np.fft.fftshift(fft_along_dim(np.fft.ifftshift(x)))
      return output

  @tf.function
  def tf_fftnshift_function(self, input):
      fft = tf.numpy_function(self.fftnshift, [input], tf.complex128)
      return fft

  def padop(self, inputs):
      x_pos = np.random.randint(0, self.rows - self.patch_size + 1)
      y_pos = np.random.randint(0, self.cols - self.patch_size + 1)
      paddings = [[self.pad_Value - x_pos, self.pad_Value - (self.rows - x_pos - self.patch_size)],
                 [self.pad_Value - y_pos, self.pad_Value - (self.cols - y_pos - self.patch_size)], [0,0]]
      output = tf.pad(inputs, paddings)
      return output

  def fftop(self, inputs):
      output = tf.map_fn(self.tf_fftnshift_function, inputs, fn_output_signature=tf.complex128)
      return output

  def taperingop(self, inputs):
      output = tf.map_fn(self.taperingfunc, inputs, fn_output_signature=(tf.complex128, tf.complex128))
      return output

  @tf.function
  def taperingfunc(self, inputs):
      win = self.create_custom_win()
      ref = tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(inputs[0]), tf.signal.fft2d(win)))
      mov = tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(inputs[1]), tf.signal.fft2d(win)))
      return ref, mov

  @tf.function
  def conv_func(self, k):
      output = tf.signal.ifft2d(tf.multiply(tf.signal.fft2d(k), tf.signal.fft2d(self.win)))
      return output

  @tf.function
  def convop(self, inputs):
      output = tf.map_fn(self.conv_func, inputs, fn_output_signature=tf.complex128)
      return output

  def regrid_func(self, k):
      output = interpn(self.gridxy, k, self.gridxyq, method='linear')
      return output

  @tf.function
  def regrid_unshuffle_tf(self, k):
      output = tf.numpy_function(self.regrid_func, [k], tf.complex128)
      output = tf.numpy_function(reshuffle, [output], tf.complex128)
      output = tf.reshape(output, (self.patch_size, self.patch_size))
      return output

  def regridop(self, inputs):
      output = tf.map_fn(self.regrid_unshuffle_tf, inputs)
      return output

  def stackop(self, inputs):
      ref_real = tf.math.real(inputs[0])
      mov_real = tf.math.real(inputs[1])
      ref_imag = tf.math.imag(inputs[0])
      mov_imag = tf.math.imag(inputs[1])
      feature1 = tf.cast(tf.stack((ref_real, ref_imag), axis=-1), tf.float32)
      feature2 = tf.cast(tf.stack((mov_real, mov_imag), axis=-1), tf.float32)
      return feature1, feature2

  def preprocess(self, input):
      output = tf.map_fn(self.padop, input)
      return output



  def call(self, input_tensor):
      if self.pad:
          x = tf.stack((input_tensor[0], input_tensor[1]), axis=-1)
          x = tf.map_fn(self.padop, x)
          x = tf.unstack(x, num=2, axis=-1)
          ref = x[0]
          mov = x[1]
          ref = self.fft_layer(ref)
          ref = self.conv_layer(ref)
          mov = self.fft_layer(mov)
          mov = self.conv_layer(mov)
      else:
          ref = input_tensor[0]
          mov = input_tensor[1]
          ref, mov = self.tapering_layer([ref, mov])

      ref = self.regrid_layer(ref)
      mov = self.regrid_layer(mov)

      x = self.stack_layer([ref, mov])

      return x


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    ModelResp = buildLAPNet_model_2D_old(33)
    print(ModelResp.summary())