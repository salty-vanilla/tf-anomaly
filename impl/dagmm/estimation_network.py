import tensorflow as tf
import os
import sys
sys.path.append('../../')
from ops.blocks import DenseBlock


class EstimationNetwork(tf.keras.Model):
    def __init__(self, dense_units,
                 normalization='batch'):
        super().__init__()
        self.denses = [DenseBlock(d,
                                  activation_=None,
                                  normalization=normalization)
                       for d in dense_units[:len(dense_units)-1]]
        self.denses += [DenseBlock(dense_units[-1],
                                   activation_='softmax',
                                   normalization=None)]

    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        for dense in self.denses:
            x = dense(x, training=training)
        return x
