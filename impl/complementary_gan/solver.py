import tensorflow as tf
import os
import sys
import time
import numpy as np
from PIL import Image
tf.enable_eager_execution()
sys.path.append('../../')
from losses import pull_away
from datasets.image_sampler import ImageSampler
from ops.losses import discriminator_loss


class Solver(object):
    def __init__(self, generator,
                 discriminator,
                 lr_g: float =1e-3,
                 lr_d: float =1e-3,
                 density_threshold: float =0.005,
                 logdir: str =None):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.train.AdamOptimizer(lr_g)
        self.opt_d = tf.train.AdamOptimizer(lr_d)
        self.latent_dim = self.generator.latent_dim
        self.thr = density_threshold
        self.eps = 1e-8
        self.logdir = logdir

    def _update_discriminator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_real = self.discriminator(x, training=True)
            d_fake = self.discriminator(gz, training=True)

            loss_d = discriminator_loss(d_real, d_fake, 'JSD')
            loss_d -= tf.reduce_mean(tf.nn.sigmoid(d_real) * tf.log(tf.nn.sigmoid(d_real)))
        grads = tape.gradient(loss_d, self.discriminator.variables)
        self.opt_d.apply_gradients(zip(grads, self.discriminator.variables))
        return loss_d

    def _update_generator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_fake, feature_fake = self.discriminator(gz, with_feature=True, training=True)
            _, feature_true = self.discriminator(x, with_feature=True, training=True)

            loss_fm = tf.reduce_mean((feature_fake - feature_true) ** 2)
            loss_kl = tf.reduce_mean(tf.boolean_mask(tf.log(tf.nn.sigmoid(d_fake)),
                                                     tf.nn.sigmoid(d_fake) > self.thr))
            loss_kl = 0 if tf.is_nan(loss_kl) else loss_kl
            loss_pt = pull_away(feature_fake)
            loss_kl += loss_pt
            loss_g = loss_kl + loss_fm
        grads = tape.gradient(loss_g, self.generator.variables)
        self.opt_g.apply_gradients(zip(grads, self.generator.variables))
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

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if x.shape[0] != batch_size:
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
                os.makedirs(os.path.join(self.logdir, 'model'), exist_ok=True)
                self.generator.save_weights(os.path.join(self.logdir, 'model', 'generator_%d.h5' % epoch))
                self.discriminator.save_weights(os.path.join(self.logdir, 'model', 'discriminator_%d.h5' % epoch))

    def _visualize(self, z, epoch, data_to_image):
        dst_path = os.path.join(self.logdir, 'image', 'epoch_%d.png' % epoch)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        outputs = self.generator(z, training=True)
        outputs = np.array(outputs)
        outputs = data_to_image(outputs)
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
