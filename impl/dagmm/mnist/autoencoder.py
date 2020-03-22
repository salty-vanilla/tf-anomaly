import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from base_autoencoder import BaseAutoEncoder as AE
from ops.blocks import ConvBlock, DenseBlock


class AutoEncoder(AE):
    def __init__(self,
                 latent_dim=2,
                 nb_filter=32,
                 last_activation='tanh',
                 normalization='batch',
                 downsampling='stride',
                 upsampling='deconv'):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         downsampling,
                         upsampling)
        self.feature_shape = (2, 2, nb_filter*(2**3))
        self.downs = [ConvBlock(nb_filter*(2**i),
                                kernel_size=(3, 3),
                                sampling=downsampling,
                                **self.conv_block_params)
                      for i in range(4)]
        self.dense_down1 = DenseBlock(64,
                                      activation_='lrelu',
                                      normalization=normalization)
        self.dense_down2 = DenseBlock(latent_dim)
        self.dense_up1 = DenseBlock(64,
                                    activation_='lrelu',
                                    normalization=normalization)
        self.dense_up2 = DenseBlock(self.feature_shape[0]*self.feature_shape[1]*self.feature_shape[2],
                                    activation_='lrelu',
                                    normalization=normalization)
        self.ups = [ConvBlock(nb_filter*(2**(3-i)),
                              kernel_size=(3, 3),
                              sampling=upsampling,
                              **self.conv_block_params)
                    for i in range(4)]
        self.last_conv = ConvBlock(1, 
                                   kernel_size=(1, 1),
                                   activation_=last_activation)

    def encode(self, inputs,
               training=None,
               mask=None):
        x = inputs
        for down in self.downs:
            x = down(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_down1(x, training=training)
        return self.dense_down2(x)

    def decode(self, inputs,
               training=None,
               mask=None):
        x = inputs
        x = self.dense_up1(x, training=training)
        x = self.dense_up2(x, training=training)
        x = tf.keras.layers.Reshape(self.feature_shape)(x)
        for up in self.ups:
            x = up(x, training=training)
        return self.last_conv(x)
