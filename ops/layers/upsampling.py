import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def temporal_upsampling(x, size=2):
    """
    Args:
        x: input_tensor (N, L)
        size: int
    Returns: tensor (N, L*ks)
    """
    with tf.variable_scope(None, temporal_upsampling.__name__):
        _x = tf.expand_dims(x, axis=2)
        _x = kl.UpSampling2D((size, 1))(_x)
        _x = tf.squeeze(_x, axis=2)
        return _x


def upsampling2d(x, size=(2, 2)):
    return kl.UpSampling2D(size)(x)