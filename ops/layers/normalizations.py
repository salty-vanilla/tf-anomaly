import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


def layer_norm(x, is_training=True):
    return tl.layer_norm(x, trainable=is_training)


def batch_norm(x, is_training=True):
    return tf.layers.batch_normalization(x,
                                         training=is_training)
    # return tl.batch_norm(x,
    #                      scale=True,
    #                      updates_collections=None,
    #                      is_training=is_training)


def pixel_norm(x, *args, **kwargs):
    return PixelNorm()(x)


class PixelNorm(Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec('float32', input_shape)

    def call(self, x, *args, **kwargs):
        return x / K.sqrt(K.mean(K.square(x),
                                 axis=-1,
                                 keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape
