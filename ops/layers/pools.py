import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def temporal_max_pool(x,
                      kernel_size=2,
                      stride=2,
                      padding='same'):
    """
    Args:
        x: input_tensor (N, L)
        kernel_size: int
        stride: int
        padding: 'same' or 'valid'
    Returns: tensor (N, L//ks)
    """
    with tf.name_scope(temporal_max_pool.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = kl.MaxPool2D((kernel_size, 1), (stride, 1), padding)(_x)
        _x = tf.squeeze(_x, axis=2)
    return _x


def temporal_average_pool(x,
                          kernel_size=2,
                          stride=2,
                          padding='same'):
    """
    Args:
        x: input_tensor (N, L)
        kernel_size: int
        stride: int
        padding: same
    Returns: tensor (N, L//ks)
    """
    with tf.name_scope(temporal_average_pool.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = kl.AveragePooling2D((kernel_size, 1), (stride, 1), padding)(_x)
        _x = tf.squeeze(_x, axis=2)
    return _x


def average_pool2d(x,
                   kernel_size=(2, 2),
                   strides=(2, 2),
                   padding='same'):
    return kl.AveragePooling2D(kernel_size, strides, padding)(x)


def max_pool2d(x,
               kernel_size=(2, 2),
               strides=(2, 2),
               padding='same'):
    return kl.MaxPool2D(kernel_size, strides, padding)(x)


def global_average_pool2d(x):
    with tf.name_scope(global_average_pool2d.__name__):
        return tf.reduce_mean(x, axis=[1, 2])
