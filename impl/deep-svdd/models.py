import tensorflow as tf


class SVDD(tf.keras.Model):
    def __init__(self, output_dim,
                 nb_filter=16,
                 normalization='batch',
                 downsampling='stride'):
        super().__init__()
        self.output_dim = output_dim
        self.nb_filter = nb_filter

        self.normalization = normalization
        self.downsampling = downsampling
        self.feature_shape = None

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization,
                                  'use_bias': False}
