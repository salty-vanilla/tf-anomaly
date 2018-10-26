import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def activation(x, func=None):
    if func == 'lrelu':
        return kl.LeakyReLU(0.2)(x)
    elif func == 'swish':
        return x * kl.Activation('sigmoid')(x)
    else:
        return kl.Activation(func)(x)