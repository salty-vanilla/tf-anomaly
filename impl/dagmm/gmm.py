import tensorflow as tf
import numpy as np


class GMMLayer(tf.keras.layers.Layer):
    def __init__(self, nb_components,
                 nb_features):
        super().__init__()

        self.nb_components = nb_components
        self.nb_features = nb_features
        self.eps = tf.keras.backend.epsilon()

    def build(self, input_shape):
        self.phi = self.add_weight('phi',
                                   (self.nb_components,),
                                   dtype=tf.float32,
                                   trainable=False,
                                   initializer=tf.keras.initializers.Zeros())

        self.mu = self.add_weight('mu',
                                  (self.nb_components, self.nb_features),
                                  dtype=tf.float32,
                                  trainable=False,
                                  initializer=tf.keras.initializers.Zeros())

        self.sigma = self.add_weight('sigma',
                                     (self.nb_components, self.nb_features, self.nb_features),
                                     dtype=tf.float32,
                                     trainable=False,
                                     initializer=tf.keras.initializers.Zeros())

        self.L = self.add_weight('L',
                                 (self.nb_components, self.nb_features, self.nb_features),
                                 dtype=tf.float32,
                                 trainable=False,
                                 initializer=tf.keras.initializers.Zeros())

    def call(self, inputs,
             training=None,
             mask=None):
        z, gamma = inputs

        # Update parameters
        if training:
            phi, mu, sigma, L = self._update(inputs)
            return self.compute_energy([z, gamma, phi, mu, sigma, L])
        else:
            return self.compute_energy([z, gamma, self.phi, self.mu, self.sigma, self.L])

    def _update(self, inputs):
        z, gamma = inputs

        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:, None]
        z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - mu[None, :, :])
        sigma = tf.einsum('ikl,ikm->klm',
                          z_centered,
                          z_centered)
        sigma /= gamma_sum[:, None, None]
        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[1]
        min_vals = tf.linalg.diag(tf.ones(n_features, dtype=tf.float32)) * self.eps
        L = tf.linalg.cholesky(sigma + min_vals[None, :, :])

        self.phi.assign(phi)
        self.mu.assign(mu)
        self.sigma.assign(sigma)
        self.L.assign(L)

        return phi, mu, sigma, L

    def compute_energy(self, inputs):
        z, _, phi, mu, _, L = inputs

        z_centered = z[:, None, :] - mu[None, :, :]  # ikl
        v = tf.linalg.triangular_solve(L,
                                       tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp" (different from orginal paper)
        d = z.get_shape().as_list()[1]
        logits = tf.math.log(phi[:, None]) \
                 - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                          + d * tf.math.log(2.0 * np.pi) + log_det_sigma[:, None])
        energies = -tf.reduce_logsumexp(logits, axis=0)

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
                              training=training)

    def cov_diag_loss(self):
        return tf.reduce_sum(tf.divide(1., tf.linalg.diag_part(self.gmm_layer.sigma)))
