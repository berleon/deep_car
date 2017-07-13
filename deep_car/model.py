from __future__ import division
import tensorflow as tf
import numpy as np


def log_p_x(x, pi_logits, means, s, delta=np.pi / (2.*180), min=-np.pi/2, max=np.pi/2):
    """
    bs - batch_size
    w - width
    k - number of components

    x: (bs, w, 1)
    mu, s, pi_logits: (bs, w, k)
    """

    # rv = tf.contrib.distributions.Logistic(means, s)
    cdf = lambda t: tf.nn.sigmoid((t - means) / s)
    p = cdf(x + delta) - cdf(x - delta)
    k = pi_logits.get_shape()[-1]
    x = tf.tile(x, [1, 1, int(k)])
    p = tf.where(
        x - delta <= min,
        cdf(min),
        p,
    )
    p = tf.where(
        x + delta >= max,
        1. - cdf(max),
        p,
    )
    return tf.reduce_logsumexp(tf.nn.log_softmax(pi_logits) + tf.log(tf.maximum(p, 1e-12)), axis=-1)


def log_p_x_pdf(x, pi_logits, means, s):
    rv = tf.contrib.distributions.Logistic(means, s)
    p = rv.prob(x)
    return tf.reduce_logsumexp(tf.nn.log_softmax(pi_logits) + tf.log(tf.maximum(p, 1e-12)), axis=-1)


def discretize_mixture(pi_logits, means, s, min=-np.pi/2, max=np.pi/2, num=180):
    x = tf.linspace(min, max, num)
    x = tf.expand_dims(x, axis=0)
    x = tf.tile(x, [tf.shape(means)[0], 1])
    x = tf.expand_dims(x, axis=-1)

    pi_logits = tf.tile(pi_logits, [1, num, 1])
    means = tf.tile(means, [1, num, 1])
    s = tf.tile(s, [1, num, 1])
    return log_p_x(
        x, pi_logits, means, s, min=min, max=max
    )


def create_inputs(input_size, len_steering_hist):
    x = tf.placeholder(tf.float32, shape=[None, input_size[1], input_size[0], 1],
                       name='image')
    steering_hist = tf.placeholder(tf.float32, shape=[None, len_steering_hist],
                                   name='steering_hist')
    return x, steering_hist


def get_model(x, steering_hist, len_steering_delta, y_delta_buckets, n_mixtures=6, reuse=False):
    with tf.variable_scope("deep_car", reuse=False):
        y_abs_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_abs_true')
        y_delta_true = tf.placeholder(tf.int32, shape=[None, len_steering_delta],
                                      name='y_delta_true')
        y_delta_true_hot = tf.one_hot(y_delta_true, depth=y_delta_buckets)

        f = 12
        k = 3

        # data images are 48x64
        l = tf.layers.conv2d(x, f, 5, padding='same')
        l = tf.layers.batch_normalization(l)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, 2, 2)

        # 24x32 after pooling
        l = tf.layers.conv2d(x, 2*f, k, padding='same')
        l = tf.layers.batch_normalization(l)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, 2, 2)

        # 12x16
        l = tf.layers.conv2d(x, 4*f, k, padding='same')
        l = tf.layers.batch_normalization(l)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, (2, 2), 2)

        # 6x8
        l = tf.layers.conv2d(x, 8*f, k, padding='same')
        l = tf.layers.batch_normalization(l)
        l = tf.nn.relu(l)
        l = tf.layers.average_pooling2d(l, (6, 8), 1)
        l = tf.contrib.layers.flatten(l)
        l = tf.concat([l, steering_hist], axis=-1)
        l = tf.layers.dense(l, 4*f)
        l = tf.layers.batch_normalization(l)
        l = tf.nn.relu(l)

        mu = np.pi * tf.tanh(tf.layers.dense(l, n_mixtures))
        pi_logits = tf.layers.dense(l, n_mixtures)
        s = tf.nn.softplus(tf.layers.dense(l, n_mixtures))

        mu = tf.expand_dims(mu, axis=1)
        pi_logits = tf.expand_dims(pi_logits, axis=1)
        s = tf.expand_dims(s, axis=1)

        y_delta_logits = tf.layers.dense(l, len_steering_delta * y_delta_buckets)
        y_delta_logits = tf.reshape(y_delta_logits, (-1, len_steering_delta, y_delta_buckets))
        y_delta_prob = tf.nn.softmax(y_delta_logits)

        y_delta_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_delta_true_hot, logits=y_delta_logits)
        y_abs_loss = -log_p_x(tf.expand_dims(y_abs_true, axis=1), pi_logits, mu, s)

        y_abs_discr_prob = discretize_mixture(pi_logits, mu, s)

        opt = tf.train.AdamOptimizer(learning_rate=0.0003)
        opt_op = opt.minimize(y_delta_loss + y_abs_loss)

        return y_abs_true, y_delta_true, opt_op, \
            y_abs_discr_prob, y_abs_loss, y_delta_prob, y_delta_loss
