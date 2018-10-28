import tensorflow as tf
import sys
import time
tf.enable_eager_execution()
sys.path.append('../../')


class ScoreComputer(object):
    def __init__(self, generator,
                 discriminator,
                 lambda_=0.1,
                 lr=1e-3):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        self.latent_dim = self.generator.latent_dim
        self.lambda_ = lambda_

    def compute_score_on_single(self, x,
                                noise_sampler,
                                iteration=5):
        _x = tf.constant(x, dtype=tf.float32)
        z = noise_sampler(1, self.latent_dim)
        z = tf.Variable(z, dtype=tf.float32)
        for _ in range(iteration):
            with tf.GradientTape() as tape:
                gz = self.generator(z, training=False)
                loss_rec = tf.reduce_mean(tf.abs(_x - gz))

                _, feature_real = self.discriminator(_x,
                                                     training=False,
                                                     with_feature=True)
                _, feature_fake = self.discriminator(gz,
                                                     training=False,
                                                     with_feature=True)
                loss_fm = tf.reduce_mean(tf.abs(feature_real - feature_fake))

                score = (1.-self.lambda_)*loss_rec + self.lambda_*loss_fm
            grads = tape.gradient(score, [z])
            z.apply_gradients(zip(grads, [z]))
        return score

    def compute_score(self, x,
                      noise_sampler,
                      iteration=5):
        scores = []
        start = time.time()
        for i, _x in enumerate(x):
            __x = _x[None, :]
            score = self.compute_score_on_single(__x,
                                                 noise_sampler,
                                                 iteration)
            scores.append(score)
            print('{} / {}  {:.1f}[s]\r'
                  .format(i, len(x), time.time() - start, end=''))
        return scores
