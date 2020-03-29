import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Discriminator as D
from ops.blocks import ConvBlock, DenseBlock


class Discriminator(D):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__(nb_filter,
                         normalization,
                         downsampling,
                         spectral_norm)

        self.convs = []
        for i in range(4):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=downsampling,
                                        **self.conv_block_params))
        self.dense = DenseBlock(1, spectral_norm=spectral_norm)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x
