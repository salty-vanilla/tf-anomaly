import tensorflow as tf


class DAGMM(tf.keras.Model):
    def __init__(self, autoencoder,
                 estimation_network,
                 gmm):
        super().__init__()
        self.autoencoder = autoencoder
        self.estimation_network = estimation_network
        self.gmm = gmm

    def call(self, inputs,
             training=None,
             mask=None,
             with_diff=False):
        z = self.autoencoder.encode(inputs,
                                    training=training)
        y = self.autoencoder.decode(z,
                                    training=training)

        diff = tf.reduce_mean(tf.square(inputs - y),
                              axis=[1, 2, 3])
        z_diff = tf.concat([z, diff[:, None]], axis=-1)

        gamma = self.estimation_network(z_diff,
                                        training=training)

        energy = self.gmm([z_diff, gamma],
                          training=training)

        if with_diff:
            return energy, diff
        else:
            return energy
