import tensorflow as tf
import os
import sys
import yaml
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
sys.path.append('../../')
from datasets.noise_sampler import NoiseSampler
from mnist.generator import Generator
from mnist.discriminator import Discriminator
from datasets.mnist import load_data
from score_computer import ScoreComputer


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)

    noise_sampler = NoiseSampler('normal')

    inlier_classes = config['train_data_params']['labels']
    outlier_classes = [i for i in range(10) if i not in inlier_classes]

    x, y = load_data('test',
                     with_label=True,
                     normalization='tanh')
    y = np.array(y)

    generator = Generator(**config['generator_params'])
    discriminator = Discriminator(**config['discriminator_params'])

    generator.build(input_shape=(None, generator.latent_dim))
    discriminator.build(input_shape=(None, 32, 32, 1))

    generator.load_weights(os.path.join(config['logdir'], 'model', 'generator_%d' % config['test_epoch']))
    discriminator.load_weights(os.path.join(config['logdir'], 'model', 'discriminator_%d' % config['test_epoch']))

    computer = ScoreComputer(generator,
                             discriminator,
                             **config['score_computer_params'])
    outputs = computer.compute_score(x, noise_sampler,
                                     **config['compute_params'])

    inlier_outputs = np.zeros(shape=(0, ))
    for c in inlier_classes:
        inlier_outputs = np.append(inlier_outputs,
                                   outputs[y == c])
    df_inlier = pd.DataFrame({'score': inlier_outputs,
                              'label': 'inlier'})

    outlier_outputs = np.zeros(shape=(0, ))
    for c in outlier_classes:
        outlier_outputs = np.append(outlier_outputs,
                                    outputs[y == c])
    df_outlier = pd.DataFrame({'score': outlier_outputs,
                               'label': 'outlier'})

    df = pd.concat([df_inlier, df_outlier], axis=0)
    df.to_csv(os.path.join(config['logdir'], 'outputs.csv'), index=None)


if __name__ == '__main__':
    main()

