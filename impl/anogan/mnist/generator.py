import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Generator as G
from ops.blocks import ConvBlock


class Generator(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv'):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling)

        self.convs = []
        for i in range(4):
            _nb_filter = nb_filter*(2**(3-i))
            self.convs.append(ConvBlock(_nb_filter,
                                        **self.conv_block_params))

            if i != 0:
                self.convs.append(ConvBlock(_nb_filter,
                                            sampling=upsampling,
                                            **self.conv_block_params))
        self.last_conv = ConvBlock(1,
                                   kernel_size=(1, 1),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = tf.reshape(inputs, (-1, 1, 1, self.latent_dim))
        x = tf.keras.layers.UpSampling2D((4, 4))(x)
        for conv in self.convs:
            x = conv(x, training=training)
        x = self.last_conv(x, training=training)
        return x
