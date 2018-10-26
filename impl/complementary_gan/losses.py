import tensorflow as tf


def pull_away(x, eps=1e-8):
    n, d = x.get_shape().as_list()
    i = tf.eye(n)
    inv_i = tf.cast(tf.cast(i - 1, tf.bool), tf.float32)

    denominator = tf.norm(x, axis=-1)[:, None]
    denominator *= tf.norm(x, axis=-1)[None, :]

    numerator = tf.reduce_sum(x[None, :]*x[:, None], axis=-1) * inv_i

    return tf.reduce_sum((numerator/(denominator+eps))**2) / (n*(n-1))
