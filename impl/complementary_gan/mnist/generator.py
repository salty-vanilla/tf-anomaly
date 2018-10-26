import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Generator as G
from ops.blocks import ConvBlock


class Generator(G):
    def __init__(self, latent_dim,
                 nb_filter=32,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv'):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling)

        self.conv1 = ConvBlock(nb_filter*8,
                               kernel_size=(3, 3),
                               **self.conv_block_params)
        self.conv2 = ConvBlock(nb_filter*4,
                               kernel_size=(3, 3),
                               sampling=upsampling,
                               **self.conv_block_params)
        self.conv3 = ConvBlock(nb_filter*2,
                               kernel_size=(3, 3),
                               sampling=upsampling,
                               **self.conv_block_params)
        self.conv4 = ConvBlock(nb_filter*1,
                               kernel_size=(3, 3),
                               sampling=upsampling,
                               **self.conv_block_params)
        self.last_conv = ConvBlock(1,
                                   kernel_size=(1, 1),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = tf.reshape(inputs, (-1, 1, 1, self.latent_dim))
        x = tf.keras.layers.UpSampling2D((4, 4))(x)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.last_conv(x, training=training)
        return x
