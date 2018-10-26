import numpy as np
from tensorflow.python.keras.datasets import mnist
from scipy.misc import imresize


def load_data(phase='train',
              target_size=(32, 32),
              normalization=None,
              with_label=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert phase in ['train', 'test']

    x = x_train if phase == 'train' else x_test
    y = y_train if phase == 'train' else y_test

    if normalization is not None:
        x = x.astype('float32')
        if normalization == 'sigmoid':
            x /= 255
        elif normalization == 'tanh':
            x = (x/255 - 0.5) * 2
        else:
            raise ValueError

    if target_size is not None or target_size != (28, 28):
        x = np.array([imresize(arr, target_size) for arr in x])

    x = np.expand_dims(x, -1)

    if with_label:
        return x, y
    else:
        return x


def load_specific_data(labels,
                       phase='train',
                       target_size=(32, 32),
                       normalization=None,
                       with_label=False):
    x, y = load_data(phase,
                     target_size,
                     normalization,
                     with_label=True)

    new_x = np.empty((0, *target_size, 1))
    new_y = np.empty((0, ))

    for l in labels:
        new_x = np.append(new_x, x[y == l], axis=0)
        new_y = np.append(new_y, y[y == l], axis=0)

    if with_label:
        return new_x, new_y
    else:
        return new_x
