import tensorflow as tf
import os
import sys
import time
from PIL import Image
import numpy as np
tf.enable_eager_execution()
sys.path.append('../../')
from datasets.image_sampler import ImageSampler
from ops.losses import discriminator_loss, generator_loss, gradient_penalty, discriminator_norm


class GANSolver(object):
    def __init__(self, generator,
                 discriminator,
                 gp_lambda: float=10.,
                 d_norm_eps: float=1e-3,
                 lr_g: float=1e-4,
                 lr_d: float=1e-4,
                 logdir: str = None):
        self.generator = generator
        self.discriminator = discriminator
        self.gp_lambda = gp_lambda
        self.d_norm_eps = d_norm_eps
        self.opt_g = tf.train.AdamOptimizer(lr_g, beta1=0.5, beta2=0.9)
        self.opt_d = tf.train.AdamOptimizer(lr_d, beta1=0.5, beta2=0.9)
        self.latent_dim = self.generator.latent_dim
        self.logdir = logdir

    def _update_discriminator(self, x, z):
        gz = self.generator(z, training=True)
        with tf.GradientTape() as tape:
            gp = gradient_penalty(self.discriminator,
                                  real=x,
                                  fake=gz)
            gp *= self.gp_lambda
        grads_gp = tape.gradient(gp, self.discriminator.variables)

        with tf.GradientTape() as tape:
            d_real = self.discriminator(x, training=True)
            d_fake = self.discriminator(gz, training=True)
            loss_d = discriminator_loss(d_real, d_fake, 'WD')
            d_norm = discriminator_norm(d_real)
            loss = loss_d + self.d_norm_eps*d_norm

        grads = tape.gradient(loss, self.discriminator.variables)
        grads = [g + ggp for g, ggp in zip(grads, grads_gp)
                 if ggp is not None]
        self.opt_d.apply_gradients(zip(grads, self.discriminator.variables))
        return loss_d, d_norm

    def _update_generator(self, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z, training=True)
            d_fake = self.discriminator(gz, training=True)
            loss_g = generator_loss(d_fake, 'WD')
        grads = tape.gradient(loss_g, self.generator.variables)
        self.opt_g.apply_gradients(zip(grads, self.generator.variables))
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
                loss_d, d_norm = self._update_discriminator(x, z)

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
