import tensorflow as tf
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('agg')
sys.path.append(os.getcwd())
sys.path.append('../../')
from datasets.noise_sampler import NoiseSampler
from mnist.generator import Generator
from mnist.discriminator import Discriminator
from datasets.mnist import load_data
from score_computer import ScoreComputer


red = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)


def plot_gen_score(original, gens, scores, dst_path):
    fig = plt.figure(figsize=(10, 3))
    ims = []
    for i, g in enumerate(gens):
        plt.subplot(1, 3, 1)
        plt.xticks([])
        plt.yticks([])
        im1 = plt.imshow(np.squeeze(original), vmin=-1., vmax=1., cmap='gray')
        plt.subplot(1, 3, 2)
        plt.xticks([])
        plt.yticks([])
        im2 = plt.imshow(g, vmin=-1., vmax=1., cmap='gray')
        ax = plt.subplot(1, 3, 3)
        im3, = plt.plot(np.arange(i+1), scores[:i+1], 
                        c=red)
        ax.yaxis.tick_right()
        ims.append([im1, im2, im3])
    writer = animation.ImageMagickWriter()
    ani = animation.ArtistAnimation(fig, ims, 
                                    interval=50, 
                                    blit=True,
                                    repeat_delay=10000)
    ani.save(dst_path, writer=writer)


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

    inliers = np.zeros(shape=(0, *x.shape[1:]))
    for c in inlier_classes:
        inliers = np.append(inliers,
                            x[y == c],
                            axis=0)
    outliers = np.zeros(shape=(0, *x.shape[1:]))
    for c in outlier_classes:
        outliers = np.append(outliers,
                             x[y == c],
                             axis=0)

    inlier = inliers[np.random.randint(len(inliers))]
    outlier = outliers[np.random.randint(len(outliers))]

    computer = ScoreComputer(generator,
                             discriminator,
                             **config['score_computer_params'])
    _, inlier_history = computer.compute_score(inlier[None, :, :, :], 
                                               noise_sampler,
                                               **config['compute_params'],
                                               with_history=True)
    _, outlier_history = computer.compute_score(outlier[None, :, :, :], 
                                                noise_sampler,
                                                **config['compute_params'],
                                                with_history=True)
    inlier_scores = np.array(inlier_history['score']).squeeze()
    inlier_gens = np.array(inlier_history['generated']).squeeze()
    inlier_z = np.array(inlier_history['z']).squeeze()
    outlier_scores = np.array(outlier_history['score']).squeeze()
    outlier_gens = np.array(outlier_history['generated']).squeeze()
    outlier_z = np.array(outlier_history['z']).squeeze()

    plot_gen_score(inlier, inlier_gens, inlier_scores, 'inlier.gif')
    plot_gen_score(outlier, outlier_gens, outlier_scores, 'outlier.gif')


if __name__ == '__main__':
    main()
