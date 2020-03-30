import tensorflow as tf
import sys
import time
import numpy as np
sys.path.append('../../')


class ScoreComputer(object):
    def __init__(self, generator,
                 discriminator,
                 lambda_=0.1,
                 lr=1e-3):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = tf.keras.optimizers.SGD(lr)
        self.latent_dim = self.generator.latent_dim
        self.lambda_ = lambda_

    def compute_score_on_batch(self, x,
                               noise_sampler,
                               iteration=5,
                               with_history=False):
        _x = tf.constant(x, dtype=tf.float32)
        z = noise_sampler(len(x), self.latent_dim)
        z = tf.Variable(z, dtype=tf.float32, 
                        trainable=True)
        history = {'score': [], 'generated': [], 'z': []}
    
        for _ in range(iteration):
            with tf.GradientTape() as tape:
                reduce_dims = tf.range(1, tf.rank(_x))
                gz = self.generator(z, training=False)
                loss_rec = tf.reduce_mean(tf.abs(_x - gz), 
                                          axis=reduce_dims)

                _, feature_real = self.discriminator(_x,
                                                     training=False,
                                                     with_feature=True)
                _, feature_fake = self.discriminator(gz,
                                                     training=False,
                                                     with_feature=True)
                reduce_dims = tf.range(1, tf.rank(feature_fake))
                loss_fm = tf.reduce_mean(tf.abs(feature_real - feature_fake), axis=reduce_dims)

                score = (1.-self.lambda_)*loss_rec + self.lambda_*loss_fm
    
            grads = tape.gradient(score, [z])
            self.optimizer.apply_gradients(zip(grads, [z]))

            if with_history:
                history['score'].append(score)
                history['generated'].append(gz)
                history['z'].append(z)
        
        if with_history:
            return score, history
        else:
            return score

    def compute_score(self, x,
                      noise_sampler,
                      batch_size=32,
                      iteration=5, 
                      with_history=False):
        steps = len(x) // batch_size
        if len(x) % batch_size:
            steps += 1
        scores = np.empty((0, ))
        start = time.time()

        history = {'score': [], 'generated': [], 'z': []}

        for iter_ in range(steps):
            _x = x[iter_*batch_size: (iter_+1)*batch_size]
            if with_history:
                score, _hist = self.compute_score_on_batch(_x, noise_sampler, iteration, with_history)
                history['score'].extend(_hist['score'])
                history['generated'].extend(_hist['generated'])
                history['z'].extend(_hist['z'])
            else:
                score = self.compute_score_on_batch(_x, noise_sampler, iteration)
            scores = np.append(scores, score)
            print(f'{iter_*batch_size} / {len(x)} {time.time()-start:.1f}[s]', end='\r')
        
        if with_history:
            return scores, history
        else:
            return scores
