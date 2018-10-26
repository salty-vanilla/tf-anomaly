import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import resize_images
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def resize(x,
           target_size,
           method='bilinear'):
    if method in ['bilinear', 'BILINEAR']:
        _method = ResizeMethod.BILINEAR
    elif method in ['nearest_neighbor', 'NEAREST_NEIGHBOR']:
        _method = ResizeMethod.NEAREST_NEIGHBOR
    elif method in ['bicubic', 'BICUBIC']:
        _method = ResizeMethod.BICUBIC
    elif method in ['area', 'AREA']:
        _method = ResizeMethod.AREA
    else:
        raise ValueError
    return resize_images(x, target_size, _method)