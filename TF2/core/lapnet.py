import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Lambda, Input, AveragePooling2D, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras
from train import modified_EPE


# Create the 2D model
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
    # model.add(Activation(leaky_relu, name='act1'))
    model.add(LeakyReLU(alpha=0.1, name='act1'))
    model.add(Conv2D(filters=128,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2"))
    # model.add(Activation(leaky_relu, name='act2'))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Conv2D(filters=256,
                     kernel_size=3,
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


# Model with descendent kernel sizes
def buildLAPNet_model_2D_old(crop_size=33):
    input_shape = (crop_size, crop_size, 4)
    inputs = Input(shape=input_shape, )
    model = keras.Sequential()
    initializer = VarianceScaling(scale=2.0)
    model.add(inputs, name="input")
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
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, lr=0.0),
                  loss=[modified_EPE],
                  metrics=['accuracy'])
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
        return tf.stack((self.a * tf.gather(model_outputs[0], [0], axis=1) + (1 - self.a) * tf.gather(model_outputs[1],
                                                                                                      [0], axis=1),
                         tf.gather(model_outputs[0], [1], axis=1),
                         tf.gather(model_outputs[1], [1], axis=1)), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
