import tensorflow as tf
import numpy as np


class GMM(tf.keras.Model):
    def __init__(self, nb_components,
                 nb_features):
        super().__init__()

        self.nb_components = nb_components
        self.nb_features = nb_features

        self.phi = self.add_variable('phi',
                                     (nb_components, ),
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.zeros(),
                                     trainable=False)

        self.mu = self.add_variable('mu',
                                    (nb_components, nb_features),
                                    dtype=tf.float32,
                                    initializer=tf.keras.initializers.zeros(),
                                    trainable=False)

        self.sigma = self.add_variable('sigma',
                                       (nb_components, nb_features, nb_features),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.zeros(),
                                       trainable=False)

        self.L = self.add_variable('L',
                                   (nb_components, nb_features, nb_features),
                                   dtype=tf.float32,
                                   initializer=tf.keras.initializers.zeros(),
                                   trainable=False)

    def call(self, inputs,
             training=None,
             mask=None):
        z, gamma = inputs

        # Update parameters
        if training:
            # Calculate mu, sigma
            # i   : index of samples
            # k   : index of components
            # l,m : index of features
            gamma_sum = tf.reduce_sum(gamma, axis=0)
            self.phi = tf.reduce_mean(gamma, axis=0)
            self.mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, None]
            z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - self.mu[None, :, :])
            self.sigma = tf.einsum('ikl,ikm->klm',
                                   z_centered,
                                   z_centered)
            self.sigma /= gamma_sum[:, None, None]

            # Calculate a cholesky decomposition of covariance in advance
            n_features = z.shape[1]
            min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
            self.L = tf.cholesky(self._sigma + min_vals[None, :, :])

        z_centered = z[:, None, :] - self.mu[None, :, :]  # ikl
        v = tf.matrix_triangular_solve(self.L,
                                       tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp" (different from orginal paper)
        d = z.get_shape().as_list()[1]
        logits = tf.log(self.phi[:, None]) \
                 - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                          + d * tf.log(2.0 * np.pi) + log_det_sigma[:, None])
        energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    def cov_diag_loss(self):
        return tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(self.sigma)))
