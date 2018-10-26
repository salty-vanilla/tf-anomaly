import tensorflow as tf


def generator_loss(d_fake,
                   metrics='JSD'):
    if metrics == 'JSD':
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake),
                                                    logits=d_fake))
    elif metrics == 'WD':
        return -tf.reduce_mean(d_fake)
    else:
        raise ValueError


def discriminator_loss(d_real,
                       d_fake,
                       metrics='JSD'):
    if metrics == 'JSD':
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real),
                                                    logits=d_real))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake),
                                                    logits=d_fake))
        return real_loss + fake_loss
    elif metrics == 'WD':
        return -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
    else:
        raise ValueError


def gradient_penalty(discriminator,
                     real,
                     fake,
                     batch_size=None):
    with tf.variable_scope('GradientPenalty'):
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise ValueError
            else:
                bs = batch_size
        else:
            bs = tf.placeholder(tf.int32, shape=[])

        if len(real.get_shape().as_list()) == 4:
            epsilon = tf.random_uniform(shape=[bs, 1, 1, 1],
                                        minval=0., maxval=1.)
        elif len(real.get_shape().as_list()) == 2:
            epsilon = tf.random_uniform(shape=[bs, 1],
                                        minval=0., maxval=1.)
        else:
            raise ValueError

        differences = fake - real
        interpolates = real + (epsilon * differences)
        gradients = tf.gradients(discriminator(interpolates, reuse=True),
                                 [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gp = tf.reduce_mean(tf.square(slopes - 1.))
        if batch_size is not None:
            return gp
        else:
            return gp, bs


def discriminator_norm(d_real):
    with tf.name_scope('DiscriminatorNorm'):
        return tf.nn.l2_loss(d_real)
