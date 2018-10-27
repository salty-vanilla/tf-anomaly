import tensorflow as tf
import os
import sys
import time
tf.enable_eager_execution()
sys.path.append('../../')
from datasets.image_sampler import ImageSampler


class Solver(object):
    def __init__(self, dagmm,
                 lambda_energy: float =0.01,
                 lambda_diag: float =1e-4,
                 lr: float =1e-4,
                 logdir: str = None):
        self.dagmm = dagmm
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.lambda_energy = lambda_energy
        self.lambda_diag = lambda_diag
        self.logdir = logdir

    def fit(self, x,
            batch_size=64,
            nb_epoch=100,
            save_steps=10):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.fit_generator(image_sampler,
                           nb_epoch=nb_epoch,
                           save_steps=save_steps)

    def fit_generator(self, image_sampler,
                      nb_epoch=100,
                      save_steps=10):
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
                with tf.GradientTape() as tape:
                    x = tf.constant(x, dtype=tf.float32)
                    energy, diff = self.dagmm(x,
                                              training=True,
                                              with_diff=True)
                    diag_loss = self.dagmm.gmm.cov_diag_loss()
                    energy = tf.reduce_mean(energy)
                    diff = tf.reduce_mean(diff)

                    loss = diff + self.lambda_energy*energy + self.lambda_diag*diag_loss
                grads = tape.gradient(loss, self.dagmm.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.dagmm.trainable_variables))

                print('iter : {} / {}  {:.1f}[s]  diff : {:.4f}  energy : {:.4f}  diag : {:.4f} \r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              diff, energy, diag_loss), end='')

            if epoch % save_steps == 0:
                os.makedirs(os.path.join(self.logdir, 'model'), exist_ok=True)
                self.dagmm.save_weights(os.path.join(self.logdir, 'model', 'dagmm_%d.h5' % epoch))
