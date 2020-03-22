import tensorflow as tf
import os
import sys
import yaml
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
sys.path.append('../../')
from datasets.mnist import load_data
from mnist.autoencoder import AutoEncoder
from estimation_network import EstimationNetwork
from gmm import GMM
from dagmm import DAGMM


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)

    inlier_classes = config['train_data_params']['labels']
    outlier_classes = [i for i in range(10) if i not in inlier_classes]

    x, y = load_data('test',
                     normalization='tanh',
                     with_label=True)
    y = np.array(y)
    x = tf.constant(x, dtype=tf.float32)

    autoencoder = AutoEncoder(**config['autoencoder_params'])
    estimation_network = EstimationNetwork(**config['estimator_params'])
    gmm = GMM(config['estimator_params']['dense_units'][-1],
              config['autoencoder_params']['latent_dim']+1)

    autoencoder.build(input_shape=(1, 32, 32, 1))
    estimation_network.build(input_shape=(1, config['autoencoder_params']['latent_dim']+1))

    dagmm = DAGMM(autoencoder,
                  estimation_network,
                  gmm)

    dagmm.load_weights(os.path.join(config['logdir'], 'model', 'dagmm_%d.h5' % config['test_epoch']))
    outputs = dagmm(x, training=False)
    outputs = np.squeeze(np.asarray(outputs))
    inlier_outputs = np.zeros(shape=(0, ))
    for c in inlier_classes:
        inlier_outputs = np.append(inlier_outputs,
                                   outputs[y == c])
    df_inlier = pd.DataFrame({'energy': inlier_outputs,
                              'label': 'inlier'})

    outlier_outputs = np.zeros(shape=(0, ))
    for c in outlier_classes:
        outlier_outputs = np.append(outlier_outputs,
                                    outputs[y == c])
    df_outlier = pd.DataFrame({'energy': outlier_outputs,
                               'label': 'outlier'})

    df = pd.concat([df_inlier, df_outlier], axis=0)
    df.to_csv(os.path.join(config['logdir'], 'outputs.csv'), index=None)


if __name__ == '__main__':
    main()
