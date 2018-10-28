import numpy as np


class NoiseSampler:
    def __init__(self, mode='uniform',
                 seed=None):
        self.mode = mode
        self.seed = seed

    def __call__(self, batch_size, noise_dim):
        if self.seed is not None:
            np.random.seed(seed=self.seed)
        if self.mode == 'uniform':
            return np.random.uniform(-1., 1.,
                                     size=(batch_size, noise_dim))
        elif self.mode == 'normal':
            return np.random.normal(0., 1., size=(batch_size, noise_dim))