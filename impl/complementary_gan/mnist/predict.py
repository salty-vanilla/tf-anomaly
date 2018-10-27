import tensorflow as tf
import os
import sys
import yaml
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
sys.path.append('../../')
tf.enable_eager_execution()
from mnist.discriminator import Discriminator
from datasets.mnist import load_data


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)

    inlier_classes = config['train_data_params']['labels']
    outlier_classes = [i for i in range(10) if i not in inlier_classes]

    x, y = load_data('test',
                     with_label=True,
                     normalization='tanh')
    y = np.array(y)

    discriminator = Discriminator(**config['discriminator_params'])
    _ = tf.nn.sigmoid(discriminator(tf.constant(x[0][None, :], dtype=tf.float32), training=False))
    discriminator.load_weights(os.path.join(config['logdir'], 'model', 'discriminator_%d.h5' % config['test_epoch']))
    outputs = tf.nn.sigmoid(discriminator(tf.constant(x, dtype=tf.float32), training=False))

    outputs = np.squeeze(np.asarray(outputs))
    inlier_outputs = np.zeros(shape=(0, ))
    for c in inlier_classes:
        inlier_outputs = np.append(inlier_outputs,
                                   outputs[y == c])
    df_inlier = pd.DataFrame({'normality': inlier_outputs,
                              'label': 'inlier'})

    outlier_outputs = np.zeros(shape=(0, ))
    for c in outlier_classes:
        outlier_outputs = np.append(outlier_outputs,
                                    outputs[y == c])
    df_outlier = pd.DataFrame({'normality': outlier_outputs,
                               'label': 'outlier'})

    df = pd.concat([df_inlier, df_outlier], axis=0)
    df.to_csv(os.path.join(config['logdir'], 'outputs.csv'), index=None)


if __name__ == '__main__':
    main()

