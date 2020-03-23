import numpy as np
import tensorflow as tf
import os
import sys
import time
from PIL import Image
sys.path.append('../../')
from ops.losses import pull_away
from datasets.image_sampler import ImageSampler
from ops.losses import discriminator_loss, generator_loss, gradient_penalty


class GANSolver(object):
    def __init__(self, generator,
                 discriminator,
                 lr_g: float =1e-3,
                 lr_d: float =1e-3,
                 logdir: str =None):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.keras.optimizers.Adam(lr_g)
        self.opt_d = tf.keras.optimizers.Adam(lr_d)
        self.latent_dim = self.generator.latent_dim
        self.logdir = logdir

    def _update_discriminator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_real = self.discriminator(x, training=True)
            d_fake = self.discriminator(gz, training=True)

            loss_d = discriminator_loss(d_real, d_fake, 'JSD')
        grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        return loss_d

    def _update_generator(self, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_fake = self.discriminator(gz, training=True)
            loss_g = generator_loss(d_fake, 'JSD')

        grads = tape.gradient(loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss_g

    def fit(self, x,
            noise_sampler,
            batch_size=64,
            nb_epoch=100,
            visualize_steps=1,
            save_steps=1):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.fit_generator(image_sampler,
                           noise_sampler,
                           nb_epoch=nb_epoch,
                           visualize_steps=visualize_steps,
                           save_steps=save_steps)

    def fit_generator(self, image_sampler,
                      noise_sampler,
                      nb_epoch=100,
                      visualize_steps=1,
                      save_steps=1):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if len(x) != batch_size:
                    continue
                z = noise_sampler(batch_size, self.latent_dim)
                # Discriminator
                x = tf.constant(x, dtype=tf.float32)
                z = tf.constant(z, dtype=tf.float32)
                loss_d = self._update_discriminator(x, z)

                # Generator
                z = noise_sampler(batch_size, self.latent_dim)
                z = tf.constant(z, dtype=tf.float32)
                loss_g = self._update_generator(z)

                print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_g : {:.4f}\r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_g), end='')

            if epoch % visualize_steps == 0:
                self._visualize(z, epoch, image_sampler.data_to_image)

            if epoch % save_steps == 0:
                os.makedirs(os.path.join(self.logdir, 'model/target'), exist_ok=True)
                self.generator.save_weights(os.path.join(self.logdir, 'model/target', 'generator_%d' % epoch))
                self.discriminator.save_weights(os.path.join(self.logdir, 'model/target', 'discriminator_%d' % epoch))

    def _visualize(self, z, epoch, data_to_image):
        dst_path = os.path.join(self.logdir, 'image/target', 'epoch_%d.png' % epoch)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        outputs = self.generator(z, training=True)
        outputs = np.array(outputs)
        outputs = data_to_image(outputs)
        if outputs.ndim == 3:
            outputs = np.expand_dims(outputs, -1)
        n, h, w, c = outputs.shape
        n_sq = int(np.sqrt(n))
        outputs = outputs[:n_sq ** 2]
        if c == 1:
            outputs = outputs.reshape(n_sq, n_sq, h, w)
            outputs = outputs.transpose(0, 2, 1, 3)
            outputs = outputs.reshape(h * n_sq, w * n_sq)
        else:
            outputs = outputs.reshape(n_sq, n_sq, h, w, 3)
            outputs = outputs.transpose(0, 2, 1, 3, 4)
            outputs = outputs.reshape(h * n_sq, w * n_sq, 3)
        Image.fromarray(outputs).save(dst_path)


class CGANSolver(object):
    def __init__(self, generator,
                 discriminator,
                 target_network,
                 lr_g: float =1e-3,
                 lr_d: float =1e-3,
                 quantile: int =5,
                 metrics='JSD',
                 d_entropy_weight=1.,
                 feature_matching='discriminator',
                 logdir: str =None,
                 threshold=None):
        self.generator = generator
        self.discriminator = discriminator
        self.target_network = target_network
        self.opt_g = tf.keras.optimizers.Adam(lr_g)
        self.opt_d = tf.keras.optimizers.Adam(lr_d)
        self.latent_dim = self.generator.latent_dim
        self.quantile = quantile
        self.metrics = metrics
        self.d_entropy_weight = d_entropy_weight
        self.feature_matching = feature_matching
        self.threshold = threshold
        self.eps = 1e-8
        self.logdir = logdir

    def _update_discriminator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_real = self.discriminator(x, training=True)
            d_fake = self.discriminator(gz, training=True)

            loss_d = discriminator_loss(d_real, d_fake, self.metrics)
            entropy = tf.reduce_mean(tf.nn.sigmoid(d_real)*tf.math.log(tf.nn.sigmoid(d_real)+self.eps))
            loss_d -= self.d_entropy_weight * entropy
        grads = tape.gradient(loss_d, self.discriminator.trainable_variables)

        if self.metrics == 'WD':
            with tf.GradientTape() as tape:
                gp = gradient_penalty(self.discriminator,
                                      real=x,
                                      fake=gz)
                gp *= 10
            grads_gp = tape.gradient(gp, self.discriminator.trainable_variables)
            grads = [g + ggp for g, ggp in zip(grads, grads_gp)
                     if ggp is not None]

        self.opt_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return loss_d

    def _update_generator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_fake, feature_fake_ = self.target_network(gz, with_feature=True, training=False)

            _, feature_fake = self.discriminator(gz, with_feature=True, training=True)
            _, feature_real = self.discriminator(x, with_feature=True, training=True)

            loss_fm = tf.reduce_mean((feature_real - feature_fake) ** 2)

            loss_kl = tf.reduce_mean(tf.boolean_mask(tf.math.log(tf.nn.sigmoid(d_fake)+self.eps),
                                                     tf.nn.sigmoid(d_fake) > self.threshold))
            loss_kl = 0 if tf.math.is_nan(loss_kl) else loss_kl
            loss_pt = pull_away(feature_fake_)
            loss_kl += loss_pt
            loss_g = loss_kl + loss_fm
        grads = tape.gradient(loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss_fm, loss_kl

    def fit(self, x,
            noise_sampler,
            batch_size=64,
            nb_epoch=100,
            visualize_steps=1,
            save_steps=1):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.fit_generator(image_sampler,
                           noise_sampler,
                           nb_epoch=nb_epoch,
                           visualize_steps=visualize_steps,
                           save_steps=save_steps)

    def fit_generator(self, image_sampler,
                      noise_sampler,
                      nb_epoch=100,
                      visualize_steps=1,
                      save_steps=1):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        if not self.threshold:
            print('computing threshold ......')
            self.compute_threshold_generator(image_sampler)
            print('density threshold: %.4f' % self.threshold)

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if len(x) != batch_size:
                    continue

                z = noise_sampler(batch_size, self.latent_dim)

                # Discriminator
                x = tf.constant(x, dtype=tf.float32)
                z = tf.constant(z, dtype=tf.float32)
                loss_d = self._update_discriminator(x, z)

                # Generator
                z = noise_sampler(batch_size, self.latent_dim)
                z = tf.constant(z, dtype=tf.float32)
                loss_fm, loss_kl = self._update_generator(x, z)

                print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_fm : {:.4f}  loss_kl : {:.4f} \r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_fm, loss_kl), end='')

            if epoch % visualize_steps == 0:
                self._visualize(z, epoch, image_sampler.data_to_image)

            if epoch % save_steps == 0:
                os.makedirs(os.path.join(self.logdir, 'model/ocgan'), exist_ok=True)
                self.generator.save_weights(os.path.join(self.logdir, 'model/ocgan', 'generator_%d.h5' % epoch))
                self.discriminator.save_weights(os.path.join(self.logdir, 'model/ocgan', 'discriminator_%d.h5' % epoch))

    def _visualize(self, z, epoch, data_to_image):
        dst_path = os.path.join(self.logdir, 'image/ocgan', 'epoch_%d.png' % epoch)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        outputs = self.generator(z, training=True)
        outputs = np.array(outputs)
        outputs = data_to_image(outputs)
        if outputs.ndim == 3:
            outputs = np.expand_dims(outputs, -1)
        n, h, w, c = outputs.shape
        n_sq = int(np.sqrt(n))
        outputs = outputs[:n_sq ** 2]
        if c == 1:
            outputs = outputs.reshape(n_sq, n_sq, h, w)
            outputs = outputs.transpose(0, 2, 1, 3)
            outputs = outputs.reshape(h * n_sq, w * n_sq)
        else:
            outputs = outputs.reshape(n_sq, n_sq, h, w, 3)
            outputs = outputs.transpose(0, 2, 1, 3, 4)
            outputs = outputs.reshape(h * n_sq, w * n_sq, 3)
        Image.fromarray(outputs).save(dst_path)

    def compute_threshold(self, x,
                          batch_size=64):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.compute_threshold_generator(image_sampler)

    def compute_threshold_generator(self, image_sampler):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        outputs = np.empty(shape=(0, 1))
        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
        for iter_ in range(1, steps_per_epoch + 1):
            x = image_sampler()
            o = tf.nn.sigmoid(self.target_network(x, training=False))
            outputs = np.append(outputs, np.array(o), axis=0)
        outputs = np.squeeze(outputs, axis=-1)
        outputs = np.sort(outputs)

        index = batch_size//self.quantile
        self.threshold = outputs[index]

        print()