import tensorflow as tf
import numpy as np


class BaseAutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 downsampling='stride',
                 upsampling='deconv'):
        super().__init__()
        self.latent_dim = latent_dim
        self.nb_filter = nb_filter
        self.last_activation = last_activation
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.feature_shape = None

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization}

        self.last_conv_block_params = {'kernel_initializer': 'he_normal',
                                       'activation_': self.last_activation,
                                       'normalization': None}

    def call(self, inputs,
             training=None,
             mask=None):
        return self.decode(self.encode(inputs, training), training)

    def encode(self, inputs,
               training=None,
               mask=None):
        raise NotImplementedError

    def decode(self, inputs,
               training=None,
               mask=None):
        raise NotImplementedError

    def encode_with_rec_loss(self, inputs,
                             training=None,
                             mask=None):
        z = self.encode(inputs, training)
        y = self.decode(z, training)
        diff = tf.reduce_mean((inputs - y)**2, axis=[1, 2, 3])
        z = tf.concat((z, diff), axis=-1)
        return z
