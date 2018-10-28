import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Discriminator as D
from ops.blocks import ConvBlock


class Discriminator(D):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride'):
        super().__init__(nb_filter,
                         normalization,
                         downsampling)

        self.convs = []
        for i in range(4):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        **self.conv_block_params))
            self.convs.append(ConvBlock(_nb_filter,
                                        sampling=downsampling,
                                        **self.conv_block_params))
        self.dense = tf.layers.Dense(1)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x)

        if with_feature:
            return x, feature_vector
        else:
            return x
