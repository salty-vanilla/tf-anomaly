import tensorflow as tf
import numpy as np


class GMMLayer(tf.keras.layers.Layer):
    def __init__(self, nb_components,
                 nb_features,
                 eps=1e-6):
        super().__init__()

        self.nb_components = nb_components
        self.nb_features = nb_features
        self.eps = eps

        self.phi = self.add_variable('phi',
                                     (nb_components,),
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.Zeros())

        self.mu = self.add_variable('mu',
                                    (nb_components, nb_features),
                                    dtype=tf.float32,
                                    initializer=tf.keras.initializers.Zeros())

        self.sigma = self.add_variable('sigma',
                                       (nb_components, nb_features, nb_features),
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.Zeros())

        self.L = self.add_variable('L',
                                   (nb_components, nb_features, nb_features),
                                   dtype=tf.float32,
                                   initializer=tf.keras.initializers.Zeros())

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
            tf.assign(self.phi, tf.reduce_sum(gamma, axis=0))
            tf.assign(self.mu, tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, None])
            z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - self.mu[None, :, :])
            sigma = tf.einsum('ikl,ikm->klm',
                              z_centered,
                              z_centered)
            sigma /= gamma_sum[:, None, None]
            tf.assign(self.sigma, sigma)
            # Calculate a cholesky decomposition of covariance in advance
            n_features = z.shape[1]
            min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * self.eps
            tf.assign(self.L, tf.cholesky(self.sigma + min_vals[None, :, :]))

        z_centered = z[:, None, :] - self.mu[None, :, :]  # ikl
        v = tf.matrix_triangular_solve(self.L,
                                       tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)), axis=1)

        # To calculate energies, use "log-sum-exp" (different from orginal paper)
        d = z.get_shape().as_list()[1]
        logits = tf.log(self.phi[:, None]) \
                 - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                          + d * tf.log(2.0 * np.pi) + log_det_sigma[:, None])
        energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies


class GMM(tf.keras.Model):
    def __init__(self, nb_components,
                 nb_features):
        super().__init__()

        self.nb_components = nb_components
        self.nb_features = nb_features

        self.gmm_layer = GMMLayer(nb_components,
                                  nb_features)

    def call(self, inputs,
             training=None,
             mask=None):
        return self.gmm_layer(inputs,
                              training,
                              mask)

    def cov_diag_loss(self):
        return tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(self.gmm_layer.sigma)))
