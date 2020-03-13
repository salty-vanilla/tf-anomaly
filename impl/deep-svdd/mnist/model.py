import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')
from models import DeepSVDD as _DeepSVDD
from ops.blocks import ConvBlock, DenseBlock


class DeepSVDD(_DeepSVDD):
    def __init__(self, output_dim,
                 nb_filter=16,
                 normalization='batch',
                 downsampling='stride'):
        super().__init__(output_dim,
                         nb_filter,
                         normalization,
                         downsampling)

        self.convs = [ConvBlock(nb_filter*(2**i),
                                kernel_size=(5, 5),
                                kernel_regularizer=tf.keras.regularizers.l2(1.),
                                **self.conv_block_params)
                      for i in range(3)]
        self.dense1 = DenseBlock(128,
                                 activation_='lrelu',
                                 kernel_regularizer=tf.keras.regularizers.l2(1.),
                                 use_bias=False,
                                 normalization=self.normalization)
        self.dense2 = DenseBlock(self.output_dim,
                                 kernel_regularizer=tf.keras.regularizers.l2(1.),
                                 use_bias=False)

    def call(self, inputs,
             training=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        return x