import tensorflow as tf
import os
import sys
import time
sys.path.append('../../')
from datasets.image_sampler import ImageSampler


class Solver(object):
    def __init__(self, model,
                 objective: str='one-class',
                 lambda_: float=1e-5,
                 nu: float=1e-5,
                 lr: float =1e-4,
                 logdir: str =None):
        self.model = model
        self.objective = objective
        self.lambda_ = lambda_
        self.nu = nu
        self.opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
        self.logdir = logdir
        self.center = None
        self.radius = None

    def update(self, x):
        weights = [layer.kernel if hasattr(layer, kernel)
                   for layer in self.model.layers]
        with tf.GradientTape() as tape:
            y = self.model(x, training=True)
            distances = tf.reduce_sum((y-self.center)**2, axis=-1)
            
            if self.objective == 'one-class':
                d = tf.reduce_mean(distances)
            else:
                distances -= self.radius**2
                d = self.radius**2 \
                    + (1./self.nu)*tf.reduce_mean(tf.math.maximum(
                                                    tf.zeros_like(distances), 
                                                    distances))

            l2_w = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights])
            
            loss = d + (self.lambda_/2.)*l2_w
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

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
        center = self.compute_center(image_sampler)
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
    
        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))

            if self.objective == 'soft-boundary':
                self.radius = self.compute_radius

            start = time.time()

            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if x.shape[0] != batch_size:
                    continue
            
                loss = self.update(x)

                print(f'iter : {iter_} / {steps_per_epoch}',
                      f'{time.time():.1f}[s]',
                      f'loss : {loss:.4f}',
                      end='\r')
            
            if epoch % save_steps == 0:
                os.makedirs(os.path.join(self.logdir, 'model'), exist_ok=True)
                self.model.save_weights(os.path.join(self.logdir, 'model', 'model_%d.h5' % epoch))

    # TODO
    def compute_center(self, image_sampler):
        pass

    # TODO
    def compute_radius(self, image_sampler):
        pass
