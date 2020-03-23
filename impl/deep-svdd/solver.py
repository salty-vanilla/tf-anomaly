import tensorflow as tf
import numpy as np
import os
import sys
import time
sys.path.append('../../')
from datasets.image_sampler import ImageSampler


class Solver(object):
    def __init__(self, model,
                 objective: str='one-class',
                 lambda_: float=1e-5,
                 nu: float=0.1,
                 lr: float=1e-4,
                 logdir: str=None):
        self.model = model
        self.objective = objective
        self.lambda_ = lambda_
        self.nu = nu
        self.opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
        self.logdir = logdir
        self.center = tf.Variable(np.zeros((self.model.output_dim)),
                                  dtype=tf.float32)
        self.radius = tf.Variable(0., dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(center=self.center,
                                              radius=self.radius)

    def update(self, x):
        with tf.GradientTape() as tape:
            y = self.model(x, training=True)
            distances = tf.reduce_sum((y-self.center)**2, axis=-1)
            
            if self.objective == 'one-class':
                d = tf.reduce_mean(distances)
            else:
                _distances = distances - self.radius**2
                d = self.radius**2 \
                    + (1./self.nu)*tf.reduce_mean(tf.math.maximum(
                                                    tf.zeros_like(_distances), 
                                                    _distances))

            l2_loss = tf.reduce_sum([l.losses[0] for l in self.model.layers])
            loss = d + (self.lambda_/2.)*l2_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, distances

    def fit(self, x,
            batch_size=64,
            nb_epoch=100,
            warm_up_epoch=10,
            save_steps=10):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.fit_generator(image_sampler,
                           nb_epoch=nb_epoch,
                           warm_up_epoch=warm_up_epoch,
                           save_steps=save_steps)

    def fit_generator(self, image_sampler,
                      nb_epoch=100,
                      warm_up_epoch=10,
                      save_steps=10):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        self.compute_center(image_sampler, steps_per_epoch)

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()

            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if x.shape[0] != batch_size:
                    continue
            
                loss, disances = self.update(x)

                print(f'iter : {iter_} / {steps_per_epoch}',
                      f'{time.time()-start:.1f}[s]',
                      f'loss : {loss:.4f}',
                      end='\r')
                
                if epoch > warm_up_epoch:
                    self.compute_radius(disances)
            
            if epoch % save_steps == 0:
                os.makedirs(os.path.join(self.logdir, 'model'), exist_ok=True)
                self.model.save_weights(os.path.join(self.logdir, 'model', 'model_%d' % epoch))
                self.checkpoint.write(os.path.join(self.logdir, 'model', 'center_radius_%d' % epoch))

    def compute_center(self, image_sampler, 
                       steps_per_epoch):
        print('Calculating Center ... ', end='')
        y = tf.raw_ops.Empty(shape=(0, self.model.output_dim), dtype=tf.float32)

        for _ in range(1, steps_per_epoch + 1):
            x = image_sampler()
            _y = self.model(x)
            y = tf.concat([y, _y], axis=0)

        center = tf.reduce_mean(y, axis=0)
        self.center.assign(center)
        print('[Done]')

    def compute_radius(self, distances):
        n = len(distances)
        index = int(n*(1.-self.nu))

        _distances = tf.sort(distances)
        self.radius.assign(_distances[index])
