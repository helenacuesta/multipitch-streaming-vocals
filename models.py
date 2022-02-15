import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Concatenate, Input, MaxPool2D, Dense, Permute, Flatten, Dropout, Lambda
from tensorflow.keras import Model

import tensorflow.keras.backend as K

import config_pred as config


def encoder_harmonic_noskip():


    inputs = Input(shape=config.input_shape)
    x = inputs

    # Encoding Path
    layer = 0
    for num_filter in config.unet_num_filters:

        if num_filter == 64:
            x = Conv2D(
                #input_shape=[config.batch_size, config.max_phr_len, config.num_features, 1],
                input_shape=config.input_shape,
                filters=num_filter,
                kernel_size=config.unet_vertical_kernel_size,
                padding='same',
                strides=(1, 1))(x)

            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

            layer += 1
        else:
            x = Conv2D(
                # input_shape=[config.batch_size, config.max_phr_len, config.num_features, 1],
                input_shape=config.input_shape,
                filters=num_filter,
                kernel_size=config.unet_kernel_size,
                padding='same',
                strides=config.strides)(x)

            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

            layer += 1

    return x, inputs

def decoder_harmonic_noskip(x):

    unet_num_layers = 6
    input_shape = config.input_shape #(config.batch_size, config.stft_size, config.patch_len, 1)

    # Decoding Path
    for layer in range(unet_num_layers):

        # Make sure that num_filter coincides with current layer shape
        if layer == unet_num_layers - 1:
            num_filter = 1
        else:
            # import pdb; pdb.set_trace()
            #num_filter = skip_layer.get_shape().as_list()[-1] // 2
            num_filter = config.unet_num_filters[-layer-2]

        if num_filter == 32:

            x = Conv2DTranspose(
                # input_shape=[config.batch_size,config.max_phr_len,config.num_features,1],
                input_shape=input_shape,
                filters=num_filter,
                kernel_size=config.unet_vertical_kernel_size,
                padding='same',
                strides=(1, 1))(x)

            x = BatchNormalization()(x)
        else:
            x = Conv2DTranspose(
                # input_shape=[config.batch_size,config.max_phr_len,config.num_features,1],
                input_shape=input_shape,
                filters=num_filter,
                kernel_size=config.unet_kernel_size,
                padding='same',
                strides=config.strides)(x)

            x = BatchNormalization()(x)

        # Activation function for decoder (sigmoid for last layer)
        if layer == unet_num_layers - 1:
            x = tf.keras.activations.get('sigmoid')(x)
        else:
            x = tf.keras.activations.get('relu')(x)

    return x


def encoder_harmonic(hcqt_flag):

    skip_connections = []

    if hcqt_flag:
        inputs = Input(shape=config.input_shape_3d)
    else:
        inputs = Input(shape=config.input_shape)

    x = inputs

    # Encoding Path
    layer = 0
    for num_filter in config.unet_num_filters:

        if num_filter == 64:
            x = Conv2D(
                #input_shape=[config.batch_size, config.max_phr_len, config.num_features, 1],
                input_shape=config.input_shape,
                filters=num_filter,
                kernel_size=config.unet_vertical_kernel_size,
                padding='same',
                strides=(1, 1))(x)

            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

            # Save skip connection for decoding path
            skip_connections.append(x)
            layer += 1
        else:
            x = Conv2D(
                # input_shape=[config.batch_size, config.max_phr_len, config.num_features, 1],
                input_shape=config.input_shape,
                filters=num_filter,
                kernel_size=config.unet_kernel_size,
                padding='same',
                strides=config.strides)(x)

            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

            # Save skip connection for decoding path
            skip_connections.append(x)
            layer += 1

    return x, skip_connections, inputs

def decoder_harmonic(x, skip_connections, hcqt_flag):

    unet_num_layers = 6

    if hcqt_flag:
        input_shape = config.input_shape_3d
    else:
        input_shape = config.input_shape #(config.batch_size, config.stft_size, config.patch_len, 1)

    # Decoding Path
    for layer in range(unet_num_layers):

        skip_layer = skip_connections[-layer - 1]

        if layer > 0:
            # import pdb; pdb.set_trace()
            x = Concatenate(axis=3)([x, skip_layer])

        # Make sure that num_filter coincides with current layer shape
        if layer == unet_num_layers - 1:
            num_filter = 1
        else:
            # import pdb; pdb.set_trace()
            num_filter = skip_layer.get_shape().as_list()[-1] // 2

        if num_filter == 32:

            x = Conv2DTranspose(
                # input_shape=[config.batch_size,config.max_phr_len,config.num_features,1],
                input_shape=input_shape,
                filters=num_filter,
                kernel_size=config.unet_vertical_kernel_size,
                padding='same',
                strides=(1, 1))(x)

            x = BatchNormalization()(x)
        else:
            x = Conv2DTranspose(
                # input_shape=[config.batch_size,config.max_phr_len,config.num_features,1],
                input_shape=input_shape,
                filters=num_filter,
                kernel_size=config.unet_kernel_size,
                padding='same',
                strides=config.strides)(x)

            x = BatchNormalization()(x)

        # Activation function for decoder (sigmoid for last layer)
        if layer == unet_num_layers - 1:
            x = tf.keras.activations.get('sigmoid')(x)
        else:
            x = tf.keras.activations.get('relu')(x)

    return x

def encoder():

    skip_connections = []

    inputs = Input(shape=config.input_shape)
    x = inputs

    # Encoding Path
    layer = 0
    for num_filter in config.unet_num_filters:
        x = Conv2D(
            #input_shape=[config.batch_size, config.max_phr_len, config.num_features, 1],
            input_shape=config.input_shape,
            filters=num_filter,
            kernel_size=config.unet_kernel_size,
            padding='same',
            strides=config.strides)(x)

        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Save skip connection for decoding path
        skip_connections.append(x)
        layer += 1

    return x, skip_connections, inputs


def decoder(x, skip_connections):

    unet_num_layers = 6
    unet_kernel_size = [5, 5]
    input_shape = config.input_shape #(config.batch_size, config.stft_size, config.patch_len, 1)

    # Decoding Path
    for layer in range(unet_num_layers):

        skip_layer = skip_connections[-layer - 1]

        if layer > 0:
            #import pdb; pdb.set_trace()
            x = Concatenate(axis=3)([x, skip_layer])

        # Make sure that num_filter coincides with current layer shape
        if layer == unet_num_layers - 1:
            num_filter = 1
        else:
            #import pdb; pdb.set_trace()
            num_filter = skip_layer.get_shape().as_list()[-1] // 2

        x = Conv2DTranspose(
            # input_shape=[config.batch_size,config.max_phr_len,config.num_features,1],
            input_shape=input_shape,
            filters=num_filter,
            kernel_size=unet_kernel_size,
            padding='same',
            strides=config.strides)(x)

        x = BatchNormalization()(x)

        # Activation function for decoder (sigmoid for last layer)
        if layer == unet_num_layers - 1:
            x = tf.keras.activations.get('sigmoid')(x)
        else:
            x = tf.keras.activations.get('relu')(x)

    return x


def unet():
    """
    Based on
    https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
    """

    # create encoding path
    encoding_path, skip_connections, inputs = encoder()

    num_decoders = 4
    outputs = []

    out1 = decoder(encoding_path, skip_connections)
    out2 = decoder(encoding_path, skip_connections)
    out3 = decoder(encoding_path, skip_connections)
    out4 = decoder(encoding_path, skip_connections)

    # for i in range(num_decoders):
    #
    #     # start the decoding path from the unique encoding output
    #     x = encoding_path
    #     # create decoder branch
    #     x = decoder(x, skip_connections)
    #     outputs = [outputs, x]
    outputs = [out1, out2, out3, out4]

    model = Model(inputs, outputs)

    return model

def unet_harmonic(hcqt_flag=False):
    """
    Based on
    https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
    """

    # create encoding path
    encoding_path, skip_connections, inputs = encoder_harmonic(hcqt_flag)


    out1 = decoder_harmonic(encoding_path, skip_connections, hcqt_flag)
    out2 = decoder_harmonic(encoding_path, skip_connections, hcqt_flag)
    out3 = decoder_harmonic(encoding_path, skip_connections, hcqt_flag)
    out4 = decoder_harmonic(encoding_path, skip_connections, hcqt_flag)

    out1 = tf.squeeze(out1, axis=-1)
    out2 = tf.squeeze(out2, axis=-1)
    out3 = tf.squeeze(out3, axis=-1)
    out4 = tf.squeeze(out4, axis=-1)

    # for i in range(num_decoders):
    #
    #     # start the decoding path from the unique encoding output
    #     x = encoding_path
    #     # create decoder branch
    #     x = decoder(x, skip_connections)
    #     outputs = [outputs, x]
    outputs = [out1, out2, out3, out4]

    model = Model(inputs, outputs)

    return model

def unet_harmonic_noskip():
    """
    Based on
    https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
    """

    # create encoding path
    encoding_path, inputs = encoder_harmonic_noskip()

    out1 = decoder_harmonic_noskip(encoding_path)
    out2 = decoder_harmonic_noskip(encoding_path)
    out3 = decoder_harmonic_noskip(encoding_path)
    out4 = decoder_harmonic_noskip(encoding_path)

    out1 = tf.squeeze(out1, axis=-1)
    out2 = tf.squeeze(out2, axis=-1)
    out3 = tf.squeeze(out3, axis=-1)
    out4 = tf.squeeze(out4, axis=-1)

    outputs = [out1, out2, out3, out4]

    model = Model(inputs, outputs)

    return model

def base_model(input, let):

    b1 = BatchNormalization()(input)

    # conv1
    y1 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1{}'.format(let))(b1)
    y1a = BatchNormalization()(y1)

    # conv2
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2{}'.format(let))(y1a)
    y2a = BatchNormalization()(y2)

    # conv3
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3{}'.format(let))(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4{}'.format(let))(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1{}'.format(let))(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2{}'.format(let))(y5a)
    y6a = BatchNormalization()(y6)

    return y6a, input

def latedeep():
    '''Late/Deep no phase
    '''

    input_shape_1 = (360, None, 1) # HCQT input shape

    inputs1 = Input(shape=input_shape_1)

    y6a, _ = base_model(inputs1, 'a')

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6a)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=inputs1, outputs=predictions)

    return model
