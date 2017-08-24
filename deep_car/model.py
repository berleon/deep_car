from __future__ import division
import tensorflow as tf
import numpy as np


class LogisticMixture:
    def __init__(self, pi_logit, mean, scale, min=0., max=1., num=100):
        self.pi_logit = pi_logit
        self.mean = mean
        self.scale = scale
        self.min = min
        self.max = max
        self.num = num

    def log_p(self, x):
        return log_p_x(x, self.pi_logit, self.mean, self.scale,
                       self.min, self.max, self.num)

    def log_p_pdf(self, x):
        rv = tf.contrib.distributions.Logistic(self.mean, self.scale)
        p = rv.prob(x)
        return tf.reduce_logsumexp(tf.nn.log_softmax(self.pi_logit) +
                                   tf.log(tf.maximum(p, 1e-12)), axis=-1)

    def discrete_probability_distribution(self):
        x = tf.linspace(self.min, self.max, self.num)
        x = tf.expand_dims(x, axis=0)
        x = tf.tile(x, [tf.shape(self.mean)[0], 1])
        x = tf.expand_dims(x, axis=-1)
        pi_logit = tf.tile(self.pi_logit, [1, self.num, 1])
        mean = tf.tile(self.mean, [1, self.num, 1])
        scale = tf.tile(self.scale, [1, self.num, 1])
        return log_p_x(
            x, pi_logit, mean, scale, min=self.min, max=self.max, num=self.num
        )


def log_p_x(x, pi_logits, means, s, min=-np.pi/2, max=np.pi/2, num=180):
    """
    bs - batch_size
    w - width
    k - number of components

    x: (bs, w, 1)
    mu, s, pi_logits: (bs, w, k)
    """
    delta = (max - min) / (2.*num)
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


class Model:
    def __init__(self, input_shape, n_mixtures=6, y_delta_buckets=2, reuse=False):
        self.n_mixtures = n_mixtures
        with tf.variable_scope("deep_car", reuse=reuse):
            # inputs
            self.image = tf.placeholder(tf.float32, shape=input_shape, name='image')
            self.steering_abs = tf.placeholder(tf.float32, shape=[None, 1], name='steering_abs')
            self.y_delta_buckets = y_delta_buckets
            # training flag for batch norm
            self.training = tf.placeholder_with_default(False, shape=[], name='training')
            # ground truth
            self.y_distance_true = tf.placeholder(
                    tf.float32, shape=[None, 1], name='y_distance_true')
            self.y_delta_true = tf.placeholder(tf.int32, shape=[None, 1], name='y_delta_true')
            self.y_delta_true_hot = tf.one_hot(self.y_delta_true, depth=y_delta_buckets)

            self.setup_distance_prob()
            self.setup_steering()
            self.setup_optimizer()

    def setup_distance_prob(self):
        f = 48
        k = 5
        # TODO: fix batch normalization

        # data images are 48x64
        l = tf.layers.conv2d(self.image, f, 5, padding='same')
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, 2, 2)

        # 24x32 after pooling
        l = tf.layers.conv2d(l, 2*f, k, padding='same')
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, 2, 2)

        print(l.get_shape())
        # 12x16
        l = tf.layers.conv2d(l, 4*f, k, padding='same')
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, (2, 2), 2)

        l = tf.layers.conv2d(l, 4*f, k, padding='same')
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, (2, 2), 2)

        print(l.get_shape())
        # 6x8
        l = tf.layers.conv2d(l, 8*f, k, padding='same')
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        l = tf.layers.average_pooling2d(l, (3, 4), 1)
        l = tf.contrib.layers.flatten(l)
        l = tf.concat([l, self.steering_abs], axis=-1)
        l = tf.layers.dense(l, 4*f)
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)

        print(l.get_shape())
        self.nn_features = l

        mu = .6*tf.tanh(tf.layers.dense(l, self.n_mixtures)) + 0.5
        pi_logits = tf.layers.dense(l, self.n_mixtures)
        s = 0.2*tf.nn.softplus(tf.layers.dense(l, self.n_mixtures))

        mu = tf.expand_dims(mu, axis=1)
        pi_logit = tf.expand_dims(pi_logits, axis=1)
        s = tf.expand_dims(s, axis=1)
        self.mixture = LogisticMixture(pi_logit, mu, s)

        self.y_distance_prob = tf.identity(
            self.mixture.discrete_probability_distribution(), name='y_distance_discr_prob')

    def setup_optimizer(self):
        self.y_distance_loss = tf.identity(
            -self.mixture.log_p(tf.expand_dims(self.y_distance_true, axis=1)),
            name='y_distance_loss')

        self.y_delta_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_delta_true_hot, logits=self.y_steering_logits,
            name='y_delta_loss')

        opt = tf.train.AdamOptimizer(learning_rate=0.0003)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt_op = opt.minimize(
                self.y_delta_loss + self.y_distance_loss, name='opt')

    def setup_steering(self):
        f = 16
        l = tf.concat([self.nn_features, self.y_distance_true], axis=-1)
        l = tf.layers.dense(l, 4*f)
        l = tf.layers.batch_normalization(l, training=self.training)
        l = tf.nn.relu(l)
        self.y_steering_logits = tf.layers.dense(l, 2)
        self.y_steering_prob = tf.nn.softmax(self.y_steering_logits,
                                             name='y_steering_prob')
