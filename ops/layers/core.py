import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl
from ops.layers import activation


def dense(x,
          units,
          activation_=None,
          use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          **kwargs):
    return activation(kl.Dense(units,
                               activation=None,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               **kwargs)(x),
                      activation_)


def reshape(x, target_shape):
    return kl.Reshape(target_shape)(x)


def flatten(x):
    return kl.Flatten()(x)


def dropout(x,
            rate=0.5,
            is_training=True):
    return tl.dropout(x, 1.-rate,
                      is_training=is_training)


def pad(x,
        size,
        mode='constant'):
    if mode in ['CONSTANT', 'constant', 'zero', 'ZERO']:
        _mode = 'CONSTANT'
    elif mode in ['REFLECT', 'reflect']:
        _mode = 'REFLECT'
    elif mode in ['SYMMETRIC', 'symmetric']:
        _mode = 'SYMMETRIC'
    else:
        raise ValueError
    return tf.pad(x,
                  [[0, 0],
                   [size[1], size[1]],
                   [size[0], size[0]],
                   [0, 0]],
                  mode=_mode)