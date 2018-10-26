import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Generator as D
from ops.blocks import ConvBlock


class Discriminator(D):
    def __init__(self, nb_filter=32,
                 normalization='batch',
                 downsampling='stride'):
        super().__init__(nb_filter,
                         normalization,
                         downsampling)

        self.conv1 = ConvBlock(nb_filter,
                               kernel_size=(3, 3),
                               sampling=downsampling,
                               **self.conv_block_params)
        self.conv2 = ConvBlock(nb_filter*2,
                               kernel_size=(3, 3),
                               sampling=downsampling,
                               **self.conv_block_params)
        self.conv3 = ConvBlock(nb_filter*4,
                               kernel_size=(3, 3),
                               sampling=downsampling,
                               **self.conv_block_params)
        self.conv4 = ConvBlock(nb_filter*8,
                               kernel_size=(3, 3),
                               sampling=downsampling,
                               **self.conv_block_params)
        self.dense = tf.layers.Dense(1)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x)

        if with_feature:
            return x, feature_vector
        else:
            return x
