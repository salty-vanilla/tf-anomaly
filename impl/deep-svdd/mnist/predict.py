import tensorflow as tf
import os
import sys
import yaml
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
sys.path.append('../../')
from datasets.mnist import load_data
from mnist.model import DeepSVDD


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

    model = DeepSVDD(**config['model_params'])
    model.build(input_shape=(None, 32, 32, 1))

    center = tf.Variable(np.zeros((model.output_dim)),
                                   dtype=tf.float32)
    radius = tf.Variable(0., dtype=tf.float32)
    checkpoint = tf.train.Checkpoint(center=center, radius=radius)

    model.load_weights(os.path.join(config['logdir'], 'model', 'model_%d' % config['test_epoch']))
    checkpoint.restore(os.path.join(config['logdir'], 'model', 'center_radius_30-3'))

    outputs = model.predict(x)
    df_inlier = create_dataframe(x, y, 
                                 center,
                                 outputs, 
                                 inlier_classes, 
                                 model.output_dim, 
                                 'inlier')
    df_outlier = create_dataframe(x, y, 
                                  center,
                                  outputs, 
                                  outlier_classes, 
                                  model.output_dim, 
                                  'outlier')
    df = pd.concat([df_inlier, df_outlier], axis=0)
    df.to_csv(os.path.join(config['logdir'], 'outputs.csv'), index=None)
    

def create_dataframe(x, y, center, outputs, classes, output_dim, label):
    _outputs = np.zeros(shape=(0, output_dim))
    for c in classes:
        _outputs = np.append(_outputs,
                             outputs[y == c], axis=0)
    distances = np.linalg.norm(_outputs - center, ord=2, axis=-1)
    df = pd.DataFrame({'label': label, 'distance': distances})
    for i in range(output_dim):
        df = df.assign(**{f'z_{i}': _outputs[:, i]})
    return df


if __name__ == '__main__':
    main()