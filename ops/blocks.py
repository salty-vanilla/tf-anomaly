import tensorflow as tf
from ops.layers.activations import activation


class ConvBlock(tf.keras.Model):
    def __init__(self, filters,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization=None,
                 **conv_params):
        super().__init__()

        assert sampling in ['deconv', 'up', 'stride', 'max_pool', 'avg_pool', 'same']

        stride = 1 if sampling in ['same', 'subpixel', 'max_pool', 'avg_pool', ] \
            else 2
        if 'stride' in conv_params:
            stride = conv_params['stride']

        # Convolution
        if sampling in ['up', 'max_pool', 'avg_pool', 'same', 'stride']:
            s = stride if sampling == 'stride' else 1
            self.conv = tf.keras.layers.Conv2D(filters,
                                               kernel_size,
                                               strides=s,
                                               padding='same',
                                               dilation_rate=dilation_rate,
                                               activation=None,
                                               **conv_params)
        elif sampling == 'deconv':
            self.conv = tf.keras.layers.Conv2DTranspose(filters,
                                                        kernel_size,
                                                        strides=stride,
                                                        padding='same',
                                                        dilation_rate=dilation_rate,
                                                        activation=None,
                                                        **conv_params)
        elif sampling == 'subpixel':
            raise NotImplementedError
        else:
            raise ValueError

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                self.norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                raise NotImplementedError
            else:
                raise ValueError
        else:
            self.norm = None

        self.act = activation_

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        return x
