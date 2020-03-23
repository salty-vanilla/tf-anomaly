import os
import sys
import shutil
import yaml
import tensorflow as tf
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import GANSolver, CGANSolver
from datasets.noise_sampler import NoiseSampler
from datasets.mnist import load_specific_data
from mnist.generator import Generator
from mnist.discriminator import Discriminator
from mnist.target import TargetNetwork


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    x = load_specific_data(phase='train',
                           **config['train_data_params'])
    noise_sampler = NoiseSampler('normal')

    generator = Generator(**config['generator_params'])
    discriminator = TargetNetwork(**config['target_params'])

    generator_ = Generator(**config['generator_params'])
    target = TargetNetwork(**config['target_params'])

    if config['target_network']:
        print('loading target .....')
        target(tf.random.normal(shape=(1, 32, 32, 1)), training=False)
        target.load_weights(config['target_network'])
    else:
        gan_solver = GANSolver(generator_,
                               target,
                               **config['gan_solver_params'],
                               logdir=config['logdir'])
        gan_solver.fit(x,
                       noise_sampler,
                       **config['gan_fit_params'])

    ocgan_solver = CGANSolver(generator,
                              discriminator,
                              target,
                              **config['ocgan_solver_params'],
                              logdir=config['logdir'])
    ocgan_solver.fit(x,
                     noise_sampler,
                     **config['ocgan_fit_params'])


if __name__ == '__main__':
    main()
