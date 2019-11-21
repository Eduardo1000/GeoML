import tensorflow as tf


def create_input(width, height, n_channels):
    # image input
    x = tf.placeholder( tf.float32, shape = (None, width, height, n_channels), name = 'X' )
    # integer class output
    y = tf.placeholder( tf.int64, shape = (None,), name = 'Y' )
    # input learning rate
    lr_placeholder = tf.placeholder( tf.float32 )

    return x, y, lr_placeholder


def blockD_2(input, initializer, num_maps1=2, num_maps2=6) :
    conv_aux = tf.layers.conv2d(inputs=input, filters=num_maps2, kernel_size=[1, 1],
                                strides=1, activation=tf.nn.relu, padding='VALID',
                                kernel_initializer=initializer)

    conv_aux = squeeze_excite_block_2(conv_aux, initializer)

    conv1 = tf.layers.conv2d(inputs=input, filters=num_maps1, kernel_size=[3, 3],
                             strides=1, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)

    conv2 = tf.layers.conv2d(inputs=input, filters=num_maps1, kernel_size=[5, 5],
                             strides=1, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)

    concat = tf.concat([conv1, conv2], 3)

    conv5 = tf.layers.conv2d(inputs=concat, filters=num_maps2, kernel_size=[3, 3], strides=1,
                             activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)

    return conv5 + conv_aux, None


def squeeze_excite_block_2(input_x, initializer, ratio=2):
    out_dim = input_x.shape.as_list()[-1]

    with tf.name_scope("squeeze"):
        squeeze = tf.reduce_mean(input_x, axis=[1,2])

        excitation = Fully_connected(squeeze, units=out_dim / ratio, initializer=initializer,
                                     layer_name='squeeze_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, initializer=initializer,
                                     layer_name='squeeze_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        return input_x * excitation


def Fully_connected(x, units, initializer, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units, kernel_initializer = initializer)


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Global_Average_Pooling(x):
    input_shape = x.shape.as_list()
    squeeze = tf.layers.average_pooling2d(x, pool_size = [input_shape[1], input_shape[2]], strides = [1, 1] )
    return squeeze
