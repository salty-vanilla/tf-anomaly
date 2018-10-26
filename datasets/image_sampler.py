from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.utils import to_categorical
import os
import numpy as np


class ImageSampler:
    def __init__(self, normalize_mode='tanh',
                 is_training=True):
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow(self, x,
             y=None,
             batch_size=32,
             shuffle=True,
             seed=None):
        return ArrayIterator(x,
                             y,
                             batch_size=batch_size,
                             normalize_mode=self.normalize_mode,
                             shuffle=shuffle,
                             seed=seed,
                             is_training=self.is_training)


class ArrayIterator(Iterator):
    def __init__(self, x,
                 y,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training):
        self.x = x
        self.y = y
        self.nb_sample = len(self.x)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.is_training = is_training

        if len(self.x.shape) == 4:
            if x.shape[3] == 1:
                self.x = self.x.reshape(self.x.shape[:3])

    def __call__(self, *args, **kwargs):
        if self.is_training:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        data_batch = np.array([preprocessing(x, self.normalize_mode)
                               for x in self.x[index_array]])
        if len(data_batch.shape) == 3:
            data_batch = np.expand_dims(data_batch, -1)
        if self.y is not None:
            label_batch = self.y[index_array]
            return data_batch, label_batch
        else:
            return data_batch

    def flow_on_test(self):
        indexes = np.arange(self.nb_sample)
        if self.shuffle:
            print('Now you set is_training = False. \nBut shuffle = True')
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1
        for i in range(steps):
            index_array = indexes[i * self.batch_size: (i + 1) * self.batch_size]
            data_batch = np.array([preprocessing(x, self.normalize_mode)
                                   for x in self.x[index_array]])
            if self.y is not None:
                label_batch = self.y[index_array]
                yield data_batch, label_batch
            else:
                yield data_batch

    def random_sampling(self, batch_size=16, seed=None):
        indexes = np.arange(self.nb_sample)
        if seed is not None:
            np.random.seed(seed=seed)
        np.random.shuffle(indexes)

        index_array = indexes[:batch_size]

        data_batch = np.array([preprocessing(x, self.normalize_mode)
                               for x in self.x[index_array]])
        if len(data_batch.shape) == 3:
            data_batch = np.expand_dims(data_batch, -1)

        if self.y is not None:
            label_batch = self.y[index_array]
            return data_batch, label_batch
        else:
            return data_batch

    def data_to_image(self, x):
        return denormalize(x, self.normalize_mode)


def preprocessing(x, normalize_mode='tanh'):
    _x = np.asarray(x)
    _x = normalize(_x, normalize_mode)
    return _x


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (normalize(x, 'sigmoid') - 0.5) * 2.
    elif mode == 'sigmoid':
        _x = x.astype('float32')
        _x /= 255
        return _x
    elif mode is None:
        return x
    else:
        raise NotImplementedError


def denormalize(x, mode='tanh'):
    if mode == 'tanh':
        return ((x + 1.) / 2 * 255).astype('uint8')
    elif mode == 'sigmoid':
        return (x * 255).astype('uint8')
    elif mode is None:
        return x
    else:
        raise NotImplementedError


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        if 'png' in path or 'jpg' in path or 'bmp' in path:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]


def main():
    import pickle
    path = '/home/nakatsuka/workspace/dataset/johoken/insert_molding/insert_molding.pkl.gz'
    with open(path, 'rb') as f:
        ((train_x, train_y), (test_x, test_y)) = pickle.load(f)

    im = ImageSampler()
    ii = im.flow(train_x)

    x = ii()[0]

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(x)), x)
    plt.savefig('temp.png')


if __name__ == '__main__':
    main()