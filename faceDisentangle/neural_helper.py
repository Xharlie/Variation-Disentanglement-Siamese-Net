#-*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

BATCH_NORM_OPS_BASE = '_bn_update_ops'
ADV_BATCH_NORM_OPS = 'adv_bn_update_ops'
GEN_BATCH_NORM_OPS = 'gen_bn_update_ops'
DIS_BATCH_NORM_OPS = 'dis_bn_update_ops'

def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_6x6(x):
    """max_pool_6x6 downsamples a feature map by 2X."""
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],
                          strides=[1, 1, 1, 1], padding='VALID')

# def batchnormalize(inputs, name, train=True, reuse=False):
#   return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
#                                       reuse=reuse,scope=name,scale=True)

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))

def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    # collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           trainable=trainable)

def batchnormalize(x, name, train=True, reuse=False):
    phase = 'gen'
    if name.startswith('dis'):
        phase = 'adv'
    elif name.startswith('gan_dis'):
        phase = 'dis'
    with tf.variable_scope(name, reuse=reuse):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        beta = _get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer)
        gamma = _get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer)

        moving_mean = _get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = _get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, 0.9997)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, 0.9997)
        tf.add_to_collection(phase+BATCH_NORM_OPS_BASE, update_moving_mean)
        tf.add_to_collection(phase+BATCH_NORM_OPS_BASE, update_moving_variance)

        if not train:
            mean, variance = moving_mean, moving_variance

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        #x.set_shape(inputs.get_shape()) ??

        return x