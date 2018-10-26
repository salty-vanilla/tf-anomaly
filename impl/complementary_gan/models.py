import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, latent_dim,
                 nb_filter=32,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv'):
        super().__init__()
        self.latent_dim = latent_dim
        self.nb_filter = nb_filter

        self.last_activation = last_activation
        self.normalization = normalization
        self.upsampling = upsampling
        self.feature_shape = None

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization}

        self.last_conv_block_params = {'kernel_initializer': 'he_normal',
                                       'activation_': self.last_activation,
                                       'normalization': None}

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def update_ops(self):
        return [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in ops.name]


class Discriminator(tf.keras.Model):
    def __init__(self, nb_filter=32,
                 normalization='batch',
                 downsampling='stride'):
        super().__init__()
        self.nb_filter = nb_filter
        self.normalization = normalization
        self.downsampling = downsampling

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization}

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def update_ops(self):
        return [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in ops.name]
