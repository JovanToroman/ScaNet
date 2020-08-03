import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *

def resnet(input_image):

    with tf.compat.v1.variable_scope("autoencoder"):

        c1 = conv2d(input_image, 64,9)

        # residual 1

        c2 = conv2d(c1, 64, 3)

        c3 = conv2d(c2, 64, 3)

        # residual 2

        c4 = conv2d(c3, 128, 3)

        c5 = conv2d(c4, 128, 3)

        # residual 3

        c6 = conv2d(c5, 256, 3)

        c7 = conv2d(c6, 256, 3)

        # residual 4

        c8 = conv2d(c7, 128, 3)

        c9 = conv2d(c8, 128, 3)

        # Convolutional

        c10 = conv2d(c9, 64, 3)

        c11 = conv2d(c10, 64, 3)

        # Final

        enhanced = conv2d(c11, 3, 9, activation='sigmoid')

    return enhanced


def autoencoder(inputs):
    with tf.compat.v1.variable_scope("autoencoder"):

        # Encoder
        net = Conv2D(100, 9, activation = tf.nn.relu, name="conv1", padding='SAME')(inputs)
        # net = tf.compat.v1.layers.max_pooling2d(net, 2, 2, padding = 'same')
        net = Conv2D(200, 3, activation = tf.nn.relu, name="conv2", padding='SAME')(net)
        net = Conv2D(400, 3, activation=tf.nn.relu, name="conv3", padding='SAME')(net)

        # Decoder
        net = Conv2D(400, 3, activation=tf.nn.relu, name="conv4", padding='SAME')(net)
        net = Conv2D(200, 3, activation=tf.nn.relu, name="conv5", padding='SAME')(net)
        # net = tf.compat.v1.image.resize_nearest_neighbor(net, tf.constant([100, 100]))
        net = Conv2D(100, 3, activation=tf.nn.relu, name="conv6", padding='SAME')(net)
        net = Conv2D(3, 9, activation =tf.nn.sigmoid, name = 'conv7', padding='SAME')(net)

        return net


def unet(inputs):
    with tf.compat.v1.variable_scope("autoencoder"):
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='mp1')(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='mp2')(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6')(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7')(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8')(conv4)
        drop4 = Dropout(0.5, name='dp1')(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9')(drop4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10')(conv5)
        drop5 = Dropout(0.5, name='dp2')(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(drop5)
        merge6 = concatenate([drop4, up6], axis=3, name='cat1')
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv13')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv14')(conv6)
        merge7 = concatenate([conv3, up7], axis=3, name='cat2')
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv15')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv16')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv17')(
            UpSampling2D(size=(2, 2), name='up1')(conv7))
        merge8 = concatenate([conv2, up8], axis=3, name='cat3')
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv18')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv19')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv20')(
            UpSampling2D(size=(2, 2), name='up2')(conv8))
        merge9 = concatenate([conv1, up9], axis=3, name='cat4')
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(conv9)
        conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv23')(conv9)
        conv10 = Conv2D(3, 1, activation='sigmoid', name='conv24')(conv9)

        return conv10


def unet2(inputs):
    with tf.compat.v1.variable_scope("autoencoder"):
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='mp1')(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='mp2')(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6')(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7')(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8')(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9')(conv4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10')(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(conv5)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(up6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv13')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv14')(conv6)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv15')(up7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv16')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv17')(
            UpSampling2D(size=(2, 2), name='up1')(conv7))
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv18')(up8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv19')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv20')(
            UpSampling2D(size=(2, 2), name='up2')(conv8))
        merge9 = concatenate([conv1, up9], axis=3, name='cat4')
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(conv9)
        conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv23')(conv9)
        conv10 = Conv2D(3, 1, activation='sigmoid', name='conv24')(conv9)

        return conv10

def weight_variable(shape, name):

    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(input, filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal'):
    return Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(input)


def _instance_norm(net):

    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

