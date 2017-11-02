# -*- coding: utf-8 -*-
import sys

sys.path.append("./")
import tensorflow as tf
import numpy as np
from neural_helper import *


class VDSN_FACE(object):
    def __init__(
            self,
            batch_size=100,
            image_shape=[96, 96, 3],
            dim_y=10,
            dim_11_fltr=32,
            dim_12_fltr=64,
            dim_21_fltr=64,
            dim_22_fltr=64,
            dim_23_fltr=128,
            dim_31_fltr=128,
            dim_32_fltr=96,
            dim_33_fltr=192,
            dim_41_fltr=192,
            dim_42_fltr=128,
            dim_43_fltr=256,
            dim_51_fltr=256,
            dim_52_fltr=160,
            dim_53_fltr=320,
            dim_FC=512,
            dim_F_I=256,
            simple_discriminator=True,
            simple_generator=True,
            simple_classifier=True,
            disentangle_obj_func='hybrid'
    ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_FC = dim_FC
        self.dim_F_I = dim_F_I
        self.dim_F_V = dim_FC - dim_F_I
        self.simple_discriminator = simple_discriminator
        self.simple_generator = simple_generator
        self.simple_classifier = simple_classifier
        # disentangle_obj_func = negative_log (-logD(x)), one_minus(log(1-D(x))) or hybrid
        self.disentangle_obj_func = disentangle_obj_func

        if not self.simple_generator:
            self.gen_W1 = tf.Variable(tf.random_normal([dim_W1, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1, dim_W2 * 7 * 7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5, 5, image_shape[-1], dim_W3], stddev=0.02), name='gen_W4')

        if not self.simple_discriminator:
            self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02),
                                          name='discrim_W1')
            self.discrim_b1 = bias_variable([self.dim_F_V], name='dis_b1')
        self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        self.discrim_b2 = bias_variable([self.dim_y], name='dis_b2')

        #  weight of encoder:
        self.encoder_W11 = tf.Variable(tf.random_normal([3, 3, image_shape[-1], dim_11_fltr], stddev=0.02), name='encoder_W11')
        self.encoder_W12 = tf.Variable(tf.random_normal([3, 3, dim_11_fltr, dim_12_fltr], stddev=0.02), name='encoder_W12')
        self.encoder_W21 = tf.Variable(tf.random_normal([3, 3, dim_12_fltr, dim_21_fltr], stddev=0.02), name='encoder_W21')
        self.encoder_W22 = tf.Variable(tf.random_normal([3, 3, dim_21_fltr, dim_22_fltr], stddev=0.02), name='encoder_W22')
        self.encoder_W23 = tf.Variable(tf.random_normal([3, 3, dim_22_fltr, dim_23_fltr], stddev=0.02), name='encoder_W23')
        self.encoder_W31 = tf.Variable(tf.random_normal([3, 3, dim_23_fltr, dim_31_fltr], stddev=0.02), name='encoder_W31')
        self.encoder_W32 = tf.Variable(tf.random_normal([3, 3, dim_31_fltr, dim_32_fltr], stddev=0.02), name='encoder_W32')
        self.encoder_W33 = tf.Variable(tf.random_normal([3, 3, dim_32_fltr, dim_33_fltr], stddev=0.02), name='encoder_W33')
        self.encoder_W41 = tf.Variable(tf.random_normal([3, 3, dim_33_fltr, dim_41_fltr], stddev=0.02), name='encoder_W41')
        self.encoder_W42 = tf.Variable(tf.random_normal([3, 3, dim_41_fltr, dim_42_fltr], stddev=0.02), name='encoder_W42')
        self.encoder_W43 = tf.Variable(tf.random_normal([3, 3, dim_42_fltr, dim_43_fltr], stddev=0.02), name='encoder_W43')
        self.encoder_W51 = tf.Variable(tf.random_normal([3, 3, dim_43_fltr, dim_51_fltr], stddev=0.02), name='encoder_W51')
        self.encoder_W52 = tf.Variable(tf.random_normal([3, 3, dim_51_fltr, dim_52_fltr], stddev=0.02), name='encoder_W52')
        self.encoder_W53 = tf.Variable(tf.random_normal([3, 3, dim_52_fltr, dim_53_fltr], stddev=0.02), name='encoder_W53')
        self.encoder_WFC = tf.Variable(tf.random_normal([dim_53_fltr, dim_FC], stddev=0.02), name='encoder_WFC')
        self.encoder_b11 = bias_variable([dim_11_fltr], name='en_b11')
        self.encoder_b12 = bias_variable([dim_12_fltr], name='en_b12')
        self.encoder_b21 = bias_variable([dim_21_fltr], name='en_b21')
        self.encoder_b22 = bias_variable([dim_22_fltr], name='en_b22')
        self.encoder_b23 = bias_variable([dim_23_fltr], name='en_b23')
        self.encoder_b31 = bias_variable([dim_31_fltr], name='en_b31')
        self.encoder_b32 = bias_variable([dim_32_fltr], name='en_b32')
        self.encoder_b33 = bias_variable([dim_33_fltr], name='en_b33')
        self.encoder_b41 = bias_variable([dim_41_fltr], name='en_b41')
        self.encoder_b42 = bias_variable([dim_42_fltr], name='en_b42')
        self.encoder_b43 = bias_variable([dim_43_fltr], name='en_b43')
        self.encoder_b51 = bias_variable([dim_51_fltr], name='en_b51')
        self.encoder_b52 = bias_variable([dim_52_fltr], name='en_b52')
        self.encoder_b53 = bias_variable([dim_53_fltr], name='en_b53')
        self.encoder_bFC = bias_variable([dim_FC], name='en_bFC')


        # Weight of generator:
        self.generator_W11 = tf.Variable(tf.random_normal([3, 3, dim_11_fltr, image_shape[-1]], stddev=0.02), name='generator_W11')
        self.generator_W12 = tf.Variable(tf.random_normal([3, 3, dim_12_fltr, dim_11_fltr], stddev=0.02), name='generator_W12')
        self.generator_W13 = tf.Variable(tf.random_normal([3, 3, dim_21_fltr, dim_12_fltr], stddev=0.02), name='generator_W13')
        self.generator_W21 = tf.Variable(tf.random_normal([3, 3, dim_22_fltr, dim_21_fltr], stddev=0.02), name='generator_W21')
        self.generator_W22 = tf.Variable(tf.random_normal([3, 3, dim_23_fltr, dim_22_fltr], stddev=0.02), name='generator_W22')
        self.generator_W23 = tf.Variable(tf.random_normal([3, 3, dim_31_fltr, dim_23_fltr], stddev=0.02), name='generator_W23')
        self.generator_W31 = tf.Variable(tf.random_normal([3, 3, dim_32_fltr, dim_31_fltr], stddev=0.02), name='generator_W31')
        self.generator_W32 = tf.Variable(tf.random_normal([3, 3, dim_33_fltr, dim_32_fltr], stddev=0.02), name='generator_W32')
        self.generator_W33 = tf.Variable(tf.random_normal([3, 3, dim_41_fltr, dim_33_fltr], stddev=0.02), name='generator_W33')
        self.generator_W41 = tf.Variable(tf.random_normal([3, 3, dim_42_fltr, dim_41_fltr], stddev=0.02), name='generator_W41')
        self.generator_W42 = tf.Variable(tf.random_normal([3, 3, dim_43_fltr, dim_42_fltr], stddev=0.02), name='generator_W42')
        self.generator_W43 = tf.Variable(tf.random_normal([3, 3, dim_51_fltr, dim_43_fltr], stddev=0.02), name='generator_W43')
        self.generator_W51 = tf.Variable(tf.random_normal([3, 3, dim_52_fltr, dim_51_fltr], stddev=0.02), name='generator_W51')
        self.generator_W52 = tf.Variable(tf.random_normal([3, 3, dim_53_fltr, dim_52_fltr], stddev=0.02), name='generator_W52')
        self.generator_WFC = tf.Variable(tf.random_normal([dim_FC, dim_53_fltr], stddev=0.02), name='generator_WFC')

        self.generator_b11 = bias_variable(image_shape[-1], name='gen_b11')
        self.generator_b12 = bias_variable([dim_11_fltr], name='gen_b12')
        self.generator_b13 = bias_variable([dim_12_fltr], name='gen_b13')
        self.generator_b21 = bias_variable([dim_21_fltr], name='gen_b21')
        self.generator_b22 = bias_variable([dim_22_fltr], name='gen_b22')
        self.generator_b23 = bias_variable([dim_23_fltr], name='gen_b23')
        self.generator_b31 = bias_variable([dim_31_fltr], name='gen_b31')
        self.generator_b32 = bias_variable([dim_32_fltr], name='gen_b32')
        self.generator_b33 = bias_variable([dim_33_fltr], name='gen_b33')
        self.generator_b41 = bias_variable([dim_41_fltr], name='gen_b41')
        self.generator_b42 = bias_variable([dim_42_fltr], name='gen_b42')
        self.generator_b43 = bias_variable([dim_43_fltr], name='gen_b43')
        self.generator_b51 = bias_variable([dim_51_fltr], name='gen_b51')
        self.generator_b52 = bias_variable([dim_52_fltr], name='gen_b52')
        self.generator_bFC = bias_variable([dim_53_fltr], name='gen_bFC')

        if not self.simple_classifier:
            self.classifier_W1 = tf.Variable(tf.random_normal([self.dim_F_I, self.dim_F_I], stddev=0.02),
                                             name='classif_W1')
            self.classifier_b1 = bias_variable([self.dim_F_V], name='cla_b1')
        self.classifier_W2 = tf.Variable(tf.random_normal([self.dim_F_I, self.dim_y], stddev=0.02), name='classif_W2')
        self.classifier_b2 = bias_variable([self.dim_y], name='cla_b2')

    def build_model(self, gen_disentangle_weight=1, gen_regularizer_weight=1,
                    dis_regularizer_weight=1, gen_cla_weight=1):

        '''
         Y for class label
        '''
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real_left = tf.placeholder(tf.float32, [None] + self.image_shape)
        image_real_right = tf.placeholder(tf.float32, [None] + self.image_shape)
        h_fc1_left = self.encoder(image_real_left)
        h_fc1_right = self.encoder(image_real_right)

        #  F_V for variance representation
        #  F_I for identity representation
        F_I_left, F_V_left = tf.split(h_fc1_left, num_or_size_splits=2, axis=1)
        F_I_right, F_V_right = tf.split(h_fc1_right, num_or_size_splits=2, axis=1)

        h4_right = self.generator(F_I_left, F_V_right)
        h4_left = self.generator(F_I_right, F_V_left)

        image_gen_left = tf.nn.sigmoid(h4_left)
        image_gen_right = tf.nn.sigmoid(h4_right)

        Y_dis_logits_left = self.discriminator(F_V_left)
        Y_dis_logits_right = self.discriminator(F_V_right)

        Y_cla_logits_left = self.classifier(F_I_left)
        Y_cla_logits_right = self.classifier(F_I_right)

        Y_dis_result_left = tf.reduce_sum(Y * tf.nn.softmax(Y_dis_logits_left), axis=1)
        Y_dis_result_right = tf.reduce_sum(Y * tf.nn.softmax(Y_dis_logits_right), axis=1)

        dis_prediction_left = [tf.reduce_max(Y_dis_result_left), tf.reduce_mean(Y_dis_result_left),
                               tf.reduce_min(Y_dis_result_left)];
        dis_prediction_right = [tf.reduce_max(Y_dis_result_right), tf.reduce_mean(Y_dis_result_right),
                                tf.reduce_min(Y_dis_result_right)];

        gen_cla_correct_prediction_left = tf.equal(tf.argmax(Y_cla_logits_left, 1), tf.argmax(Y, 1))
        gen_cla_accuracy_left = tf.reduce_mean(tf.cast(gen_cla_correct_prediction_left, tf.float32))

        gen_cla_correct_prediction_right = tf.equal(tf.argmax(Y_cla_logits_right, 1), tf.argmax(Y, 1))
        gen_cla_accuracy_right = tf.reduce_mean(tf.cast(gen_cla_correct_prediction_right, tf.float32))

        gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
        encoder_vars = filter(lambda x: x.name.startswith('encoder'), tf.trainable_variables())
        discriminator_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
        classifier_vars = filter(lambda x: x.name.startswith('classif'), tf.trainable_variables())

        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        gen_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=gen_vars + encoder_vars + classifier_vars)
        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)

        gen_recon_cost_left = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size
        gen_recon_cost_right = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size

        gen_disentangle_cost_left = self.gen_disentangle_cost(Y, Y_dis_logits_left)
        gen_disentangle_cost_right = self.gen_disentangle_cost(Y, Y_dis_logits_right)

        gen_cla_cost_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_cla_logits_left))
        gen_cla_cost_right = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_cla_logits_right))

        dis_loss_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_dis_logits_left))
        dis_loss_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_dis_logits_right))

        gen_recon_cost = (gen_recon_cost_left + gen_recon_cost_right) / 2
        gen_disentangle_cost = (gen_disentangle_cost_left + gen_disentangle_cost_right) / 2
        gen_cla_cost = (gen_cla_cost_left + gen_cla_cost_right) / 2
        gen_total_cost = gen_recon_cost \
                         + gen_disentangle_weight * gen_disentangle_cost \
                         + gen_cla_weight * gen_cla_cost \
                         + gen_regularizer_weight * gen_regularization_loss
        dis_cost_tf = (dis_loss_left + dis_loss_right) / 2
        dis_total_cost_tf = dis_cost_tf + dis_regularizer_weight * dis_regularization_loss
        gen_cla_accuracy = (gen_cla_accuracy_left + gen_cla_accuracy_right) / 2

        tf.summary.scalar('gen_recon_cost', gen_recon_cost)
        tf.summary.scalar('gen_disentangle_cost', gen_disentangle_cost)
        tf.summary.scalar('gen_total_cost', gen_total_cost)
        tf.summary.scalar('dis_cost_tf', dis_cost_tf)
        tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf)
        tf.summary.scalar('dis_prediction_max', tf.reduce_max([dis_prediction_left[0], dis_prediction_right[0]]))
        tf.summary.scalar('dis_prediction_mean', (dis_prediction_left[1] + dis_prediction_right[1]) / 2)
        tf.summary.scalar('dis_prediction_min', tf.reduce_min([(dis_prediction_left[2], dis_prediction_right[2])]))
        tf.summary.scalar('gen_cla_accuracy', gen_cla_accuracy)

        return Y, image_real_left, image_real_right, gen_recon_cost, gen_disentangle_cost, \
               gen_cla_cost, gen_total_cost, \
               dis_cost_tf, dis_total_cost_tf, image_gen_left, image_gen_right, \
               dis_prediction_left, dis_prediction_right, gen_cla_accuracy, F_I_left, F_V_left

    def gen_disentangle_cost(self, label, logits):
        minus_one_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=1 - logits))
        negative_log_loss = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=logits))
        if self.disentangle_obj_func == 'one_minus':
            return minus_one_loss
        elif self.disentangle_obj_func == 'negative_log':
            return negative_log_loss
        return (minus_one_loss + negative_log_loss) / 2

    def encoder(self, image):

        # First convolutional layer - maps one grayscale image to 64 feature maps.
        with tf.name_scope('encoder_conv11'):
            h_conv11 = lrelu(batchnormalize(
                tf.nn.conv2d(image, self.encoder_W11, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b11))
        with tf.name_scope('encoder_conv12'):
            h_conv12 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv11, self.encoder_W12, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b12))
        with tf.name_scope('encoder_conv21'):
            h_conv21 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv12, self.encoder_W21, strides=[1, 2, 2, 1], padding='SAME') + self.encoder_b21))
        with tf.name_scope('encoder_conv22'):
            h_conv22 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv21, self.encoder_W22, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b22))
        with tf.name_scope('encoder_conv23'):
            h_conv23 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv22, self.encoder_W23, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b23))
        with tf.name_scope('encoder_conv31'):
            h_conv31 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv23, self.encoder_W31, strides=[1, 2, 2, 1], padding='SAME') + self.encoder_b31))
        with tf.name_scope('encoder_conv32'):
            h_conv32 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv31, self.encoder_W32, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b32))
        with tf.name_scope('encoder_conv33'):
            h_conv33 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv32, self.encoder_W33, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b33))
        with tf.name_scope('encoder_conv41'):
            h_conv41 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv33, self.encoder_W41, strides=[1, 2, 2, 1], padding='SAME') + self.encoder_b41))
        with tf.name_scope('encoder_conv42'):
            h_conv42 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv41, self.encoder_W42, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b42))
        with tf.name_scope('encoder_conv43'):
            h_conv43 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv42, self.encoder_W43, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b43))
        with tf.name_scope('encoder_conv51'):
            h_conv51 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv43, self.encoder_W51, strides=[1, 2, 2, 1], padding='SAME') + self.encoder_b51))
        with tf.name_scope('encoder_conv52'):
            h_conv52 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv51, self.encoder_W52, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b52))
        with tf.name_scope('encoder_conv53'):
            h_conv53 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv52, self.encoder_W53, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b53))
        # ave pooling layer.
        with tf.name_scope('encoder_avg_pool'):
            h_pool= avg_pool_6x6(h_conv53)
        # Fully connected layer 320 to 512 features
        with tf.name_scope('encoder_fc'):
            h_pool_flat = tf.reshape(h_pool, [-1, self.dim_FC])
            h_fc = lrelu(batchnormalize(tf.matmul(h_pool_flat, self.encoder_WFC) + self.encoder_bFC))
        return h_fc

    def discriminator(self, F_V):
        # 512 to 512
        h1 = F_V
        if not self.simple_discriminator:
            h1 = lrelu(batchnormalize(tf.matmul(F_V, self.discrim_W1) + self.discrim_b1))
            # 512 to 10
        h2 = lrelu(batchnormalize(tf.matmul(h1, self.discrim_W2) + self.discrim_b2))
        return h2

    def generator(self, F_I, F_V):

        with tf.name_scope('gen_combine'):
            F_combine = tf.concat(axis=1, values=[F_I, F_V])
        with tf.name_scope('gen_FC'):
            h_FC = lrelu(batchnormalize(tf.matmul(F_combine, self.generator_WFC) + self.encoder_bFC))
        with tf.name_scope('gen_52'):
            output_shape = [tf.shape(h_FC)[0], 14, 14, self.dim_W3]
            h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
            h2 = tf.reshape(h2, [-1, 7, 7, self.dim_W2])

        output_shape_l3 = [tf.shape(h2)[0], 14, 14, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(batchnormalize(h3))
        output_shape_l4 = [tf.shape(h3)[0], 28, 28, self.image_shape[-1]]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        return h4

    def classifier(self, F_I):
        h1 = F_I
        if not self.simple_discriminator:
            h1 = lrelu(batchnormalize(tf.matmul(F_I, self.classifier_W1) + self.classifier_b1))
            # 512 to 10
        h2 = lrelu(batchnormalize(tf.matmul(h1, self.classifier_W2) + self.classifier_b2))
        return h2