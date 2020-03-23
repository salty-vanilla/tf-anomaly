import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Generator as G
from ops.blocks import ConvBlock, DenseBlock


class Generator(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**3),
                                activation_='relu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(1, 4):
            _nb_filter = nb_filter*(2**(3-i))

            if upsampling == 'subpixel':
                _nb_filter *= 4
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=upsampling,
                                        **self.conv_block_params))
        self.last_conv = ConvBlock(1,
                                   kernel_size=(1, 1),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**3)))
        for conv in self.convs:
            x = conv(x, training=training)
        x = self.last_conv(x, training=training)
        return x
