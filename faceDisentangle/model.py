# -*- coding: utf-8 -*-
import sys

sys.path.append("./")
import tensorflow as tf
import numpy as np
from neural_helper import *


class VDSN_FACE(object):
    def __init__(
            self,
            batch_size=64,
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
            disentangle_obj_func='hybrid'
    ):
        self.is_training = True
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_FC = dim_FC
        self.dim_F_I = dim_F_I
        self.dim_F_V = dim_FC - dim_F_I
        self.dim_53_fltr = dim_53_fltr
        # disentangle_obj_func = negative_log (-logD(x)), one_minus(log(1-D(x))) or hybrid
        self.disentangle_obj_func = disentangle_obj_func

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
        self.encoder_WFC = tf.Variable(tf.random_normal([6*6*dim_53_fltr, dim_FC], stddev=0.02), name='encoder_WFC')
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
        self.generator_W11 = tf.Variable(tf.random_normal([3, 3, image_shape[-1], dim_11_fltr], stddev=0.02), name='generator_W11')
        self.generator_W12 = tf.Variable(tf.random_normal([3, 3, dim_11_fltr, dim_12_fltr], stddev=0.02), name='generator_W12')
        self.generator_W13 = tf.Variable(tf.random_normal([3, 3, dim_12_fltr, dim_21_fltr], stddev=0.02), name='generator_W13')
        self.generator_W21 = tf.Variable(tf.random_normal([3, 3, dim_21_fltr, dim_22_fltr], stddev=0.02), name='generator_W21')
        self.generator_W22 = tf.Variable(tf.random_normal([3, 3, dim_22_fltr, dim_23_fltr], stddev=0.02), name='generator_W22')
        self.generator_W23 = tf.Variable(tf.random_normal([3, 3, dim_23_fltr, dim_31_fltr], stddev=0.02), name='generator_W23')
        self.generator_W31 = tf.Variable(tf.random_normal([3, 3, dim_31_fltr, dim_32_fltr], stddev=0.02), name='generator_W31')
        self.generator_W32 = tf.Variable(tf.random_normal([3, 3, dim_32_fltr, dim_33_fltr], stddev=0.02), name='generator_W32')
        self.generator_W33 = tf.Variable(tf.random_normal([3, 3, dim_33_fltr, dim_41_fltr], stddev=0.02), name='generator_W33')
        self.generator_W41 = tf.Variable(tf.random_normal([3, 3, dim_41_fltr, dim_42_fltr], stddev=0.02), name='generator_W41')
        self.generator_W42 = tf.Variable(tf.random_normal([3, 3, dim_42_fltr, dim_43_fltr], stddev=0.02), name='generator_W42')
        self.generator_W43 = tf.Variable(tf.random_normal([3, 3, dim_43_fltr, dim_51_fltr], stddev=0.02), name='generator_W43')
        self.generator_W51 = tf.Variable(tf.random_normal([3, 3, dim_51_fltr, dim_52_fltr], stddev=0.02), name='generator_W51')
        self.generator_W52 = tf.Variable(tf.random_normal([3, 3, dim_52_fltr, dim_53_fltr], stddev=0.02), name='generator_W52')
        self.generator_WFC = tf.Variable(tf.random_normal([dim_FC, dim_53_fltr*6*6], stddev=0.02), name='generator_WFC')

        self.generator_b11 = bias_variable([image_shape[-1]], name='gen_b11')
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
        self.generator_bFC = bias_variable([dim_53_fltr*6*6], name='gen_bFC')


        # Weight of classifier:
        self.classifier_W1 = tf.Variable(tf.random_normal([self.dim_F_I, self.dim_y], stddev=0.02), name='classif_W1')
        self.classifier_b1 = bias_variable([self.dim_y], name='cla_b1')

        # Weight of discriminator:
        self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02), name='discrim_W1')
        self.discrim_b1 = bias_variable([self.dim_F_V], name='dis_b1')
        self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        self.discrim_b2 = bias_variable([self.dim_y], name='dis_b2')

        # GAN parameters:
        self.gan_discrim_W11 = tf.Variable(tf.random_normal([3, 3, image_shape[-1], dim_11_fltr], stddev=0.02), name='gan_discrim_W11')
        self.gan_discrim_W12 = tf.Variable(tf.random_normal([3, 3, dim_11_fltr, dim_12_fltr], stddev=0.02), name='gan_discrim_W12')
        self.gan_discrim_W21 = tf.Variable(tf.random_normal([3, 3, dim_12_fltr, dim_21_fltr], stddev=0.02), name='gan_discrim_W21')
        self.gan_discrim_W22 = tf.Variable(tf.random_normal([3, 3, dim_21_fltr, dim_22_fltr], stddev=0.02), name='gan_discrim_W22')
        self.gan_discrim_W23 = tf.Variable(tf.random_normal([3, 3, dim_22_fltr, dim_23_fltr], stddev=0.02), name='gan_discrim_W23')
        self.gan_discrim_W31 = tf.Variable(tf.random_normal([3, 3, dim_23_fltr, dim_31_fltr], stddev=0.02), name='gan_discrim_W31')
        self.gan_discrim_W32 = tf.Variable(tf.random_normal([3, 3, dim_31_fltr, dim_32_fltr], stddev=0.02), name='gan_discrim_W32')
        self.gan_discrim_W33 = tf.Variable(tf.random_normal([3, 3, dim_32_fltr, dim_33_fltr], stddev=0.02), name='gan_discrim_W33')
        self.gan_discrim_W41 = tf.Variable(tf.random_normal([3, 3, dim_33_fltr, dim_41_fltr], stddev=0.02), name='gan_discrim_W41')
        self.gan_discrim_W42 = tf.Variable(tf.random_normal([3, 3, dim_41_fltr, dim_42_fltr], stddev=0.02), name='gan_discrim_W42')
        self.gan_discrim_W43 = tf.Variable(tf.random_normal([3, 3, dim_42_fltr, dim_43_fltr], stddev=0.02), name='gan_discrim_W43')
        self.gan_discrim_W51 = tf.Variable(tf.random_normal([3, 3, dim_43_fltr, dim_51_fltr], stddev=0.02), name='gan_discrim_W51')
        self.gan_discrim_W52 = tf.Variable(tf.random_normal([3, 3, dim_51_fltr, dim_52_fltr], stddev=0.02), name='gan_discrim_W52')
        self.gan_discrim_W53 = tf.Variable(tf.random_normal([3, 3, dim_52_fltr, dim_53_fltr], stddev=0.02), name='gan_discrim_W53')
        self.gan_discrim_WFC = tf.Variable(tf.random_normal([6*6*dim_53_fltr, dim_FC], stddev=0.02), name='gan_discrim_WFC')
        self.gan_discrim_WFC1 = tf.Variable(tf.random_normal([dim_FC, dim_y+1], stddev=0.02), name='gan_discrim_WFC1')
        self.gan_discrim_b11 = bias_variable([dim_11_fltr], name='gan_dis_b11')
        self.gan_discrim_b12 = bias_variable([dim_12_fltr], name='gan_dis_b12')
        self.gan_discrim_b21 = bias_variable([dim_21_fltr], name='gan_dis_b21')
        self.gan_discrim_b22 = bias_variable([dim_22_fltr], name='gan_dis_b22')
        self.gan_discrim_b23 = bias_variable([dim_23_fltr], name='gan_dis_b23')
        self.gan_discrim_b31 = bias_variable([dim_31_fltr], name='gan_dis_b31')
        self.gan_discrim_b32 = bias_variable([dim_32_fltr], name='gan_dis_b32')
        self.gan_discrim_b33 = bias_variable([dim_33_fltr], name='gan_dis_b33')
        self.gan_discrim_b41 = bias_variable([dim_41_fltr], name='gan_dis_b41')
        self.gan_discrim_b42 = bias_variable([dim_42_fltr], name='gan_dis_b42')
        self.gan_discrim_b43 = bias_variable([dim_43_fltr], name='gan_dis_b43')
        self.gan_discrim_b51 = bias_variable([dim_51_fltr], name='gan_dis_b51')
        self.gan_discrim_b52 = bias_variable([dim_52_fltr], name='gan_dis_b52')
        self.gan_discrim_b53 = bias_variable([dim_53_fltr], name='gan_dis_b53')
        self.gan_discrim_bFC = bias_variable([dim_FC], name='gan_dis_bFC')
        self.gan_discrim_bFC1 = bias_variable([dim_y+1], name='gan_dis_bFC1')

    def build_model(self, gen_disentangle_weight=1, gen_regularizer_weight=1,
                    dis_regularizer_weight=1, gen_cla_weight=1):

        '''
         Y for class label
        '''
        Y_left = tf.placeholder(tf.float32, [None, self.dim_y])
        Y_right = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real_left = tf.placeholder(tf.float32, [None] + self.image_shape)
        image_real_right = tf.placeholder(tf.float32, [None] + self.image_shape)

        F_I_left, F_V_left = self.encoder(image_real_left, reuse=False)
        F_I_right, F_V_right = self.encoder(image_real_right, reuse=True)

        h4_right = self.generator(F_I_left, F_V_right, reuse=False)
        h4_left = self.generator(F_I_right, F_V_left, reuse=True)

        image_gen_left = tf.nn.sigmoid(h4_left)
        image_gen_right = tf.nn.sigmoid(h4_right)

        Y_dis_logits_left = self.discriminator(F_V_left, reuse=False)
        Y_dis_logits_right = self.discriminator(F_V_right, reuse=True)

        Y_cla_logits_left = self.classifier(F_I_left, reuse=False)
        Y_cla_logits_right = self.classifier(F_I_right, reuse=True)

        Y_dis_result_left = tf.reduce_sum(Y_left * tf.nn.softmax(Y_dis_logits_left), axis=1)
        Y_dis_result_right = tf.reduce_sum(Y_right * tf.nn.softmax(Y_dis_logits_right), axis=1)

        dis_prediction_left = [tf.reduce_max(Y_dis_result_left), tf.reduce_mean(Y_dis_result_left),
                               tf.reduce_min(Y_dis_result_left)];
        dis_prediction_right = [tf.reduce_max(Y_dis_result_right), tf.reduce_mean(Y_dis_result_right),
                                tf.reduce_min(Y_dis_result_right)];

        gen_cla_correct_prediction_left = tf.equal(tf.argmax(Y_cla_logits_left, 1), tf.argmax(Y_left, 1))
        gen_cla_accuracy_left = tf.reduce_mean(tf.cast(gen_cla_correct_prediction_left, tf.float32))

        gen_cla_correct_prediction_right = tf.equal(tf.argmax(Y_cla_logits_right, 1), tf.argmax(Y_right, 1))
        gen_cla_accuracy_right = tf.reduce_mean(tf.cast(gen_cla_correct_prediction_right, tf.float32))

        gen_vars = filter(lambda x: x.name.startswith('generator'), tf.trainable_variables())
        encoder_vars = filter(lambda x: x.name.startswith('encoder'), tf.trainable_variables())
        discriminator_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
        classifier_vars = filter(lambda x: x.name.startswith('classif'), tf.trainable_variables())
        gan_discriminators_vars = filter(lambda x: x.name.startswith('gan_discrim'), tf.trainable_variables())

        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        gen_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=gen_vars + encoder_vars + classifier_vars)

        gan_dis_regularization_loss = tf.contrib.layers.apply_regularization(
                regularizer, weights_list=gan_discriminators_vars)

        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)

        gen_recon_cost_left = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size
        gen_recon_cost_right = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size

        gen_disentangle_cost_left = self.gen_disentangle_cost(Y_left, Y_dis_logits_left)
        gen_disentangle_cost_right = self.gen_disentangle_cost(Y_right, Y_dis_logits_right)

        gen_cla_cost_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_left, logits=Y_cla_logits_left))
        gen_cla_cost_right = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_right, logits=Y_cla_logits_right))

        dis_loss_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_left, logits=Y_dis_logits_left))
        dis_loss_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_right, logits=Y_dis_logits_right))

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

        ### GAN Loss definition
        Y_real_right = tf.concat(axis=1, values=(Y_right, tf.zeros([tf.shape(Y_right)[0], 1])))
        Y_real_left = tf.concat(axis=1, values=(Y_left, tf.zeros([tf.shape(Y_left)[0], 1])))
        gan_gen_cost = (self.GAN_discriminiator(image_gen_left, Y_real_right, reuse=False)
                        + self.GAN_discriminiator(image_gen_right, Y_real_left, reuse=True)) / 2

        Y_fake_left = tf.concat(axis=1, values=(tf.zeros([tf.shape(Y_right)[0], self.dim_y]),
                                                tf.ones([tf.shape(Y_right)[0], 1])))
        Y_fake_right = tf.concat(axis=1, values=(tf.zeros([tf.shape(Y_left)[0], self.dim_y]),
                                                 tf.ones([tf.shape(Y_left)[0], 1])))
        gan_dis_cost_gen = (self.GAN_discriminiator(image_gen_left, Y_fake_left, reuse=True)
                                + self.GAN_discriminiator(image_gen_right, Y_fake_right, reuse=True)) / 2
        gan_dis_cost_real = (self.GAN_discriminiator(image_real_left, Y_real_left, reuse=True)
                                + self.GAN_discriminiator(image_real_right, Y_real_right, reuse=True)) / 2

        gan_dis_cost = gan_dis_cost_real + gan_dis_cost_gen \
                        + gen_regularizer_weight * gan_dis_regularization_loss

        gan_total_cost = gan_gen_cost \
                        + gen_disentangle_weight * gen_disentangle_cost \
                        + gen_cla_weight * gen_cla_cost \
                        + gen_regularizer_weight * gen_regularization_loss

        tf.summary.scalar('gen_recon_cost', gen_recon_cost)
        tf.summary.scalar('gen_disentangle_cost', gen_disentangle_cost)
        tf.summary.scalar('gen_total_cost', gen_total_cost)
        tf.summary.scalar('dis_cost_tf', dis_cost_tf)
        tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf)
        tf.summary.scalar('dis_prediction_max', tf.reduce_max([dis_prediction_left[0], dis_prediction_right[0]]))
        tf.summary.scalar('dis_prediction_mean', (dis_prediction_left[1] + dis_prediction_right[1]) / 2)
        tf.summary.scalar('dis_prediction_min', tf.reduce_min([(dis_prediction_left[2], dis_prediction_right[2])]))
        tf.summary.scalar('gen_cla_accuracy', gen_cla_accuracy)
        tf.summary.scalar('gan_dis_cost', gan_dis_cost)
        tf.summary.scalar('gan_gen_cost', gan_gen_cost)
        tf.summary.scalar('gan_total_cost', gan_total_cost)

        return Y_left, Y_right, image_real_left, image_real_right, gen_recon_cost, gen_disentangle_cost, \
               gen_cla_cost, gen_total_cost, \
               dis_cost_tf, dis_total_cost_tf, image_gen_left, image_gen_right, \
               dis_prediction_left, dis_prediction_right, gen_cla_accuracy, F_I_left, F_V_left, \
               gan_gen_cost, gan_dis_cost, gan_total_cost

    def GAN_discriminiator(self, image, Y, reuse=False):
        with tf.name_scope('gan_discrim_conv11'):
            h_conv11 = lrelu(batchnormalize(
                tf.nn.conv2d(image, self.gan_discrim_W11, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b11))
                            ,'gan_dis_bn11', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv12'):
            h_conv12 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv11, self.gan_discrim_W12, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b12))
                            ,'gan_dis_bn12', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv21'):
            h_conv21 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv12, self.gan_discrim_W21, strides=[1, 2, 2, 1], padding='SAME') # + self.gan_discrim_b21))
                            ,'gan_dis_bn21', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv22'):
            h_conv22 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv21, self.gan_discrim_W22, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b22))
                            ,'gan_dis_b22', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv23'):
            h_conv23 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv22, self.gan_discrim_W23, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b23))
                            ,'gan_dis_b23', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv31'):
            h_conv31 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv23, self.gan_discrim_W31, strides=[1, 2, 2, 1], padding='SAME') # + self.gan_discrim_b31))
                            ,'gan_dis_bn31', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv32'):
            h_conv32 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv31, self.gan_discrim_W32, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b32))
                            ,'gan_dis_bn32', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv33'):
            h_conv33 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv32, self.gan_discrim_W33, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b33))
                            , 'gan_dis_bn33', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv41'):
            h_conv41 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv33, self.gan_discrim_W41, strides=[1, 2, 2, 1], padding='SAME') # + self.gan_discrim_b41))
                            ,'gan_dis_bn41', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv42'):
            h_conv42 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv41, self.gan_discrim_W42, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b42))
                            ,'gan_dis_bn42', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv43'):
            h_conv43 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv42, self.gan_discrim_W43, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b43))
                            ,'gan_dis_bn43', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv51'):
            h_conv51 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv43, self.gan_discrim_W51, strides=[1, 2, 2, 1], padding='SAME') # + self.gan_discrim_b51))
                            ,'gan_dis_bn51', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv52'):
            h_conv52 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv51, self.gan_discrim_W52, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b52))
                            ,'gan_dis_bn52', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_discrim_conv53'):
            h_conv53 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv52, self.gan_discrim_W53, strides=[1, 1, 1, 1], padding='SAME') # + self.gan_discrim_b53))
                            ,'gan_dis_bm53', train=self.is_training, reuse=reuse))
        # ave pooling layer.
        with tf.name_scope('gan_discrim_avg_pool'):
            h_pool = avg_pool_6x6(h_conv53)
        with tf.name_scope('gan_discrim_fc'):
            h_pool_flat = tf.reshape(h_pool, [-1,6*6*self.dim_53_fltr])
            h_fc = lrelu(batchnormalize(tf.matmul(h_pool_flat, self.gan_discrim_WFC) + self.gan_discrim_bFC,
                                        'gan_dis_bn', train=self.is_training, reuse=reuse))
        with tf.name_scope('gan_dis_fc2'):
            h_fc2 = tf.matmul(h_fc, self.gan_discrim_WFC1) + self.gan_discrim_bFC1

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h_fc2))
        return loss

    def gen_disentangle_cost(self, label, logits):
        p = tf.nn.softmax(logits)
        minus_one_loss = tf.reduce_mean(self.entropy_calculation(label, 1-p))
        negative_log_loss = -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=logits))
        entropy_loss = -1 * self.entropy_calculation(p,p)
        if self.disentangle_obj_func == 'one_minus':
            return minus_one_loss
        if self.disentangle_obj_func == 'negative_log':
            return negative_log_loss
        if self.disentangle_obj_func == 'entropy':
            return entropy_loss / 10
        if self.disentangle_obj_func == 'hybrid':
            return (minus_one_loss + negative_log_loss) / 2
        return (minus_one_loss + negative_log_loss) / 2 + entropy_loss / 10

    def entropy_calculation(self, p1, p2):
        p1 = tf.convert_to_tensor(p1)
        p2 = tf.convert_to_tensor(p2)
        precise_p2 = tf.cast(p2, tf.float32) if (
            p2.dtype == tf.float16) else p2
        # labels and logits must be of the same type
        p1 = tf.cast(p1, precise_p2.dtype)
        return tf.reduce_mean(-tf.reduce_sum(p1 * tf.log(p2), reduction_indices=[1]))

    def encoder(self, image, reuse=True):

        # First convolutional layer - maps one grayscale image to 64 feature maps.
        with tf.name_scope('encoder_conv11'):
            h_conv11 = lrelu(batchnormalize(
                tf.nn.conv2d(image, self.encoder_W11, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b11,
                            ,'en_bn11', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv12'):
            h_conv12 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv11, self.encoder_W12, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b12,
                            ,'en_bn12', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv21'):
            h_conv21 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv12, self.encoder_W21, strides=[1, 2, 2, 1], padding='SAME') # + self.encoder_b21,
                            ,'en_bn21', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv22'):
            h_conv22 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv21, self.encoder_W22, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b22,
                            ,'en_bn22', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv23'):
            h_conv23 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv22, self.encoder_W23, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b23,
                            ,'en_bn23', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv31'):
            h_conv31 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv23, self.encoder_W31, strides=[1, 2, 2, 1], padding='SAME') # + self.encoder_b31,
                            ,'en_bn31', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv32'):
            h_conv32 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv31, self.encoder_W32, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b32,
                            ,'en_bn32', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv33'):
            h_conv33 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv32, self.encoder_W33, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b33,
                            ,'en_bn33', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv41'):
            h_conv41 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv33, self.encoder_W41, strides=[1, 2, 2, 1], padding='SAME') # + self.encoder_b41,
                            ,'en_bn41', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv42'):
            h_conv42 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv41, self.encoder_W42, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b42,
                            ,'en_bn42', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv43'):
            h_conv43 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv42, self.encoder_W43, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b43,
                            ,'en_bn43', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv51'):
            h_conv51 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv43, self.encoder_W51, strides=[1, 2, 2, 1], padding='SAME') # + self.encoder_b51,
                            ,'en_bn51', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv52'):
            h_conv52 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv51, self.encoder_W52, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b52,
                            ,'en_bn52', train=self.is_training, reuse=reuse))
        with tf.name_scope('encoder_conv53'):
            h_conv53 = lrelu(batchnormalize(
                tf.nn.conv2d(h_conv52, self.encoder_W53, strides=[1, 1, 1, 1], padding='SAME') # + self.encoder_b53,
                            ,'en_bn53', train=self.is_training, reuse=reuse))
        # ave pooling layer.
        with tf.name_scope('encoder_avg_pool'):
            h_pool= avg_pool_6x6(h_conv53)
        # Fully connected layer 320 to 512 features
        with tf.name_scope('encoder_fc'):
            h_pool_flat = tf.reshape(h_pool, [-1, 6*6*self.dim_53_fltr])
            h_fc = lrelu(batchnormalize(tf.matmul(h_pool_flat, self.encoder_WFC),
                        'en_hfc', train=self.is_training, reuse=reuse))
        F_I, F_V = tf.split(h_fc, [self.dim_F_I, self.dim_F_V], axis = 1)
        return batchnormalize(F_I, 'en_bn3', train=self.is_training, reuse=reuse), \
               batchnormalize(F_V, 'en_bn4', train=self.is_training, reuse=reuse)

    def discriminator(self, F_V, reuse=True):
        # 512 to 512
        h1 = lrelu(batchnormalize(tf.matmul(F_V, self.discrim_W1), 'dis_bn1', train=self.is_training, reuse=reuse))
        h2 = tf.matmul(h1, self.discrim_W2) + self.discrim_b2
        return h2

    def generator(self, F_I, F_V, reuse=True):

        with tf.name_scope('gen_combine'):
            F_combine = tf.concat(axis=1, values=[F_I, F_V])
        with tf.name_scope('gen_FC'):
            h_FC = lrelu(batchnormalize(tf.matmul(F_combine, self.generator_WFC) + self.generator_bFC,
                'gen_bnFC', train=self.is_training, reuse=reuse))
            h_FC = tf.reshape(h_FC, [-1, 6, 6, self.dim_53_fltr])
        with tf.name_scope('gen_52'):
            output_shape = [tf.shape(h_FC)[0], 6, 6, self.generator_W52.shape.as_list()[2]]
            h_52 = tf.nn.conv2d_transpose(h_FC, self.generator_W52, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_52 = lrelu(batchnormalize(h_52 + self.generator_b52,
                'gen_bn52', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_51'):
            output_shape = [tf.shape(h_52)[0], 6, 6, self.generator_W51.shape.as_list()[2]]
            h_51 = tf.nn.conv2d_transpose(h_52, self.generator_W51, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_51 = lrelu(batchnormalize(h_51 + self.generator_b51,
                'gen_bn51', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_43'):
            output_shape = [tf.shape(h_51)[0], 12, 12, self.generator_W43.shape.as_list()[2]]
            h_43 = tf.nn.conv2d_transpose(h_51, self.generator_W43, output_shape=output_shape, strides=[1, 2, 2, 1])
            h_43 = lrelu(batchnormalize(h_43 + self.generator_b43,
                'gen_bn43', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_42'):
            output_shape = [tf.shape(h_43)[0], 12, 12, self.generator_W42.shape.as_list()[2]]
            h_42 = tf.nn.conv2d_transpose(h_43, self.generator_W42, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_42 = lrelu(batchnormalize(h_42 + self.generator_b42,
                'gen_bn42', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_41'):
            output_shape = [tf.shape(h_42)[0], 12, 12, self.generator_W41.shape.as_list()[2]]
            h_41 = tf.nn.conv2d_transpose(h_42, self.generator_W41, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_41 = lrelu(batchnormalize(h_41 + self.generator_b41,
                'gen_bn41', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_33'):
            output_shape = [tf.shape(h_41)[0], 24, 24, self.generator_W33.shape.as_list()[2]]
            h_33 = tf.nn.conv2d_transpose(h_41, self.generator_W33, output_shape=output_shape, strides=[1, 2, 2, 1])
            h_33 = lrelu(batchnormalize(h_33 + self.generator_b33,
                'gen_bn33', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_32'):
            output_shape = [tf.shape(h_33)[0], 24, 24, self.generator_W32.shape.as_list()[2]]
            h_32 = tf.nn.conv2d_transpose(h_33, self.generator_W32, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_32 = lrelu(batchnormalize(h_32 + self.generator_b32,
                'gen_bn32', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_31'):
            output_shape = [tf.shape(h_32)[0], 24, 24, self.generator_W31.shape.as_list()[2]]
            h_31 = tf.nn.conv2d_transpose(h_32, self.generator_W31, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_31 = lrelu(batchnormalize(h_31 + self.generator_b31,
                'gen_bn31', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_23'):
            output_shape = [tf.shape(h_31)[0], 48, 48, self.generator_W23.shape.as_list()[2]]
            h_23 = tf.nn.conv2d_transpose(h_31, self.generator_W23, output_shape=output_shape, strides=[1, 2, 2, 1])
            h_23 = lrelu(batchnormalize(h_23 + self.generator_b23,
                'gen_bn23', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_22'):
            output_shape = [tf.shape(h_23)[0], 48, 48, self.generator_W22.shape.as_list()[2]]
            h_22 = tf.nn.conv2d_transpose(h_23, self.generator_W22, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_22 = lrelu(batchnormalize(h_22 + self.generator_b22,
                'gen_bn22', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_21'):
            output_shape = [tf.shape(h_22)[0], 48, 48, self.generator_W21.shape.as_list()[2]]
            h_21 = tf.nn.conv2d_transpose(h_22, self.generator_W21, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_21 = lrelu(batchnormalize(h_21 + self.generator_b21,
                'gen_bn21', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_13'):
            output_shape = [tf.shape(h_21)[0], 96, 96, self.generator_W13.shape.as_list()[2]]
            h_13 = tf.nn.conv2d_transpose(h_21, self.generator_W13, output_shape=output_shape, strides=[1, 2, 2, 1])
            h_13 = lrelu(batchnormalize(h_13 + self.generator_b13,
                'gen_bn13', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_12'):
            output_shape = [tf.shape(h_13)[0], 96, 96, self.generator_W12.shape.as_list()[2]]
            h_12 = tf.nn.conv2d_transpose(h_13, self.generator_W12, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_12 = lrelu(batchnormalize(h_12 + self.generator_b12,
                'gen_bn12', train=self.is_training, reuse=reuse))
        with tf.name_scope('gen_11'):
            output_shape = [tf.shape(h_12)[0], 96, 96, self.generator_W11.shape.as_list()[2]]
            h_11 = tf.nn.conv2d_transpose(h_12, self.generator_W11, output_shape=output_shape, strides=[1, 1, 1, 1])
            h_11 = h_11 + self.generator_b11
        return h_11


    def classifier(self, F_I, reuse=False):
        return tf.matmul(F_I, self.classifier_W1) + self.classifier_b1
