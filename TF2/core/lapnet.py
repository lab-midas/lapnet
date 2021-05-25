import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Lambda, concatenate, Input, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras as keras
from e2eflow.core.train import modified_EPE

# Create the model
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


def buildLAPNet_model_3D():
    initializer = VarianceScaling(scale=2.0)

    input_shape = (33, 33, 4)
    coronal_input = Input(shape=input_shape, name="coronal")
    sagital_input = Input(shape=input_shape, name="sagital")
    axial_input = Input(shape=input_shape, name="axial")

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

    axial_features = Conv2D(filters=64,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv1")(axial_input)
    axial_features = LeakyReLU(alpha=0.1, name='a_act1')(axial_features)
    axial_features = Conv2D(filters=128,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv2")(axial_features)
    axial_features = LeakyReLU(alpha=0.1, name='a_act2')(axial_features)
    axial_features = Conv2D(filters=256,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv3")(axial_features)
    axial_features = LeakyReLU(alpha=0.1, name='a_act3')(axial_features)
    axial_features = Conv2D(filters=512,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv3_1")(axial_features)
    axial_features = LeakyReLU(alpha=0.1, name='a_act3_1')(axial_features)
    axial_features = Conv2D(filters=1024,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv4")(axial_features)
    axial_features = LeakyReLU(alpha=0.1, name='a_act4')(axial_features)
    axial_features = Conv2D(filters=1024,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_regularizer=l2(0.0004),
                            kernel_initializer=initializer,
                            name="a_conv4_1")(axial_features)
    axial_features = LeakyReLU(alpha=0.1, name='a_act4_1')(axial_features)
    axial_features = AveragePooling2D(pool_size=5, name='pool_a')(axial_features)
    axial_features = Conv2D(2, [1, 1], name="a_fc3")(axial_features)


    model_coronal = keras.Model(inputs=[coronal_input], outputs=[coronal_features])
    model_sagital = keras.Model(inputs=[sagital_input], outputs=[sagital_features])
    model_axial = keras.Model(inputs=[axial_input], outputs=[axial_features])

    #merged_layers = concatenate([model_coronal.output, model_sagital.output, model_axial.output])
    #x = merge([coronal_features, sagital_features, axial_features], mode='concat', axis=-1)
    #x = tf.stack([x, axial_features], axis=-1)

    #x = Conv2D(3, [1, 1], name="fc3")(merged_layers)

    #final_flow = Lambda(squeeze_func, name="fc8/squeezed")(x)

    #model = keras.Model([model_coronal.input, model_sagital.input, model_axial.input], [final_flow])

    #x = tf.keras.layers.Concatenate(axis=-1)([coronal_features, sagital_features, axial_features])
    #x = Conv2D(3, [1, 1], name="fc3")(x)
    #x = AveragePooling2D(pool_size=5, name='pool')(x)
    #final_flow = Lambda(squeeze_func, name="fc8/squeezed")(x)
    final_flow_c = Lambda(squeeze_func, name="fc8/squeezed_c")(coronal_features)
    final_flow_s = Lambda(squeeze_func, name="fc8/squeezed_s")(sagital_features)
    final_flow_a = Lambda(squeeze_func, name="fc8/squeezed_a")(axial_features)
    model = keras.Model(
        inputs=[coronal_input, sagital_input, axial_input],
        outputs=[final_flow_c, final_flow_s, final_flow_a],
    )
    return model


def squeeze_func(x):
    return tf.squeeze(x, axis=[1, 2])

def squeeze_func_3D(x):
    return tf.squeeze(x, axis=[1, 2, 3])





def load_tf1_LAPNet_cropping_ckpt(checkpoint_path_old='/mnt/data/projects/MoCo/LAPNet/UnFlow/log/ex/resp/srx424_drUS_1603/model.ckpt'):

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


def tensor_in_checkpoint_file(tensor_name):
    file_name = '/mnt/data/projects/MoCo/LAPNet/UnFlow/log/ex/resp/srx424_drUS_1603/model.ckpt'
    reader = tf.python.training.py_checkpoint_reader.NewCheckpointReader(file_name)
    res = tf.constant_initializer(reader.get_tensor(tensor_name))
    return res



