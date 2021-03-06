# -*- coding: utf-8 -*-
import sys

sys.path.append("../")
import numpy as np
from neural_helper import *
import math


class VDSN(object):
    def __init__(
            self,
            batch_size=100,
            image_shape=[28, 28, 1],
            dim_y=10,
            dim_W1=128,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=64,
            disentangle_obj_func='hybrid',
            train_bn = False,
            soft_bn = False,
            split_encoder = True
    ):
        self.is_training = True
        self.split_encoder = split_encoder
        self.train_bn = train_bn
        self.soft_bn = soft_bn
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_F_I = dim_F_I
        self.dim_F_V = dim_W1 - dim_F_I
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        # disentangle_obj_func = negative_log (-logD(x)), one_minus(log(1-D(x))) or hybrid
        self.disentangle_obj_func = disentangle_obj_func

        self.gen_W1 = tf.Variable(tf.random_normal([dim_W1, dim_W2 * 7 * 7], stddev=0.02), name='generator_W1')
        self.gen_b1 = bias_variable([self.dim_W2 * 7 * 7], name='gen_b1')
        self.gen_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name='generator_W2')
        self.gen_b2 = bias_variable([self.dim_W3], name='gen_b2')
        self.gen_W3 = tf.Variable(tf.random_normal([5, 5, image_shape[-1], dim_W3], stddev=0.02), name='generator_W3')
        self.gen_b3 = bias_variable([image_shape[-1]], name='gen_b3')

        self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02), name='discrim_W1')
        self.discrim_b1 = bias_variable([self.dim_F_V], name='dis_b1')
        self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        self.discrim_b2 = bias_variable([self.dim_y], name='dis_b2')

        self.encoder_W1 = tf.Variable(tf.random_normal([5, 5, image_shape[-1], dim_W3], stddev=0.02), name='encoder_W1')
        self.encoder_b1 = bias_variable([dim_W3], name='en_b1')
        if self.split_encoder:
            self.encoder_W2_FI = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_F_I], stddev=0.02), name='encoder_W2_FI')
            self.encoder_W2_FV = tf.Variable(tf.random_normal([5, 5, dim_W3, self.dim_W1 - self.dim_F_I], stddev=0.02), name='encoder_W2_FV')
            self.encoder_W3_FI = tf.Variable(tf.random_normal([dim_F_I * 7 * 7, self.dim_F_I], stddev=0.02), name='encoder_W3_FI')
            self.encoder_W3_FV = tf.Variable(tf.random_normal([self.dim_W1 - self.dim_F_I * 7 * 7, self.dim_W1 - self.dim_F_I], stddev=0.02), name='encoder_W3_FV')
        else:
            self.encoder_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name='encoder_W2')
            self.encoder_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7, dim_W1], stddev=0.02), name='encoder_W3')
            self.encoder_b2 = bias_variable([dim_W2], name='en_b2')
            self.encoder_b3 = bias_variable([dim_W1], name='en_b3')

        self.classifier_W1 = tf.Variable(tf.random_normal([self.dim_F_I, self.dim_y], stddev=0.02), name='classif_W1')
        self.classifier_b1 = bias_variable([self.dim_y], name='cla_b1')

        self.gan_dis_W1 = tf.Variable(tf.random_normal([5, 5, image_shape[-1], dim_W3], stddev=0.02),
                                      name='gan_discrim_W1')
        self.gan_dis_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name='gan_discrim_W2')
        self.gan_dis_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7, dim_W1], stddev=0.02), name='gan_discrim_W3')
        self.gan_dis_W4 = tf.Variable(tf.random_normal([dim_W1, dim_y + 1], stddev=0.02), name='gan_discrim_W4')
        self.gan_dis_b1 = bias_variable([dim_W3], name='gan_dis_b1')
        self.gan_dis_b2 = bias_variable([dim_W2], name='gan_dis_b2')
        self.gan_dis_b3 = bias_variable([dim_W1], name='gan_dis_b3')
        self.gan_dis_b4 = bias_variable([dim_y + 1], name='gan_dis_b4')

        self.even_label = tf.convert_to_tensor(np.ones((self.batch_size, self.dim_y)) / self.dim_y)

    def build_model(self, gen_disentangle_weight=1, gen_regularizer_weight=1,
                    dis_regularizer_weight=1, gen_cla_weight=1):

        '''
         Y for class label
        '''
        Y_left = tf.placeholder(tf.float32, [None, self.dim_y])
        Y_right = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real_left = tf.placeholder(tf.float32, [None] + self.image_shape)
        image_real_right = tf.placeholder(tf.float32, [None] + self.image_shape)
        #  F_V for variance representation
        #  F_I for identity representation
        F_I_left, F_V_left = self.encoder(image_real_left, reuse=False)
        F_I_right, F_V_right = self.encoder(image_real_right, reuse=True)

        h3_right = self.generator(F_I_left, F_V_right, reuse=False)
        h3_left = self.generator(F_I_right, F_V_left, reuse=True)

        image_gen_left = tf.nn.sigmoid(h3_left)
        image_gen_right = tf.nn.sigmoid(h3_right)

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
        gan_discriminator_vars = filter(lambda x: x.name.startswith('gan_discrim'), tf.trainable_variables())

        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        gen_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=gen_vars + encoder_vars + classifier_vars)

        gan_dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=gan_discriminator_vars)

        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)
        shape =  self.batch_size #* self.image_shape[0] * self.image_shape[1]
        gen_recon_cost_left = tf.nn.l2_loss(image_real_left - image_gen_left) / shape
        gen_recon_cost_right = tf.nn.l2_loss(image_real_right - image_gen_right) / shape

        gen_disentangle_cost_left = self.gen_disentangle_cost(Y_left, Y_dis_logits_left)
        gen_disentangle_cost_right = self.gen_disentangle_cost(Y_right, Y_dis_logits_right)

        gen_cla_cost_left = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_left, logits=Y_cla_logits_left))
        gen_cla_cost_right = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_right, logits=Y_cla_logits_right))

        dis_loss_left = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_left, logits=Y_dis_logits_left))
        dis_loss_right = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y_right, logits=Y_dis_logits_right))

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

        #### GAN LOSS
        Y_real_right = tf.concat(axis=1, values=(Y_right, tf.zeros([tf.shape(Y_right)[0], 1])))
        Y_real_left = tf.concat(axis=1, values=(Y_left, tf.zeros([tf.shape(Y_left)[0], 1])))
        gan_gen_cost = (self.GAN_discriminator(image_gen_left, Y_real_right, reuse=False)
                        + self.GAN_discriminator(image_gen_right, Y_real_left, reuse=True)) / 2
        gan_gen_FV_cost = (tf.nn.l2_loss(self.encoder(image_gen_left, reuse=True)[1] - F_V_left) + \
                          tf.nn.l2_loss(self.encoder(image_gen_right, reuse=True)[1] - F_V_right)) / self.batch_size / self.dim_F_V
        Y_fake_left = tf.concat(axis=1, values=(tf.zeros([tf.shape(Y_right)[0], self.dim_y]),
                                                tf.ones([tf.shape(Y_right)[0], 1])))
        Y_fake_right = tf.concat(axis=1, values=(tf.zeros([tf.shape(Y_left)[0], self.dim_y]),
                                                 tf.ones([tf.shape(Y_left)[0], 1])))

        gan_dis_cost_gen = (self.GAN_discriminator(image_gen_left, Y_fake_left, reuse=True)
                            + self.GAN_discriminator(image_gen_right, Y_fake_right, reuse=True)) / 2
        gan_dis_cost_real = (self.GAN_discriminator(image_real_left, Y_real_left, reuse=True)
                             + self.GAN_discriminator(image_real_right, Y_real_right, reuse=True)) / 2
        gan_dis_cost = gan_dis_cost_real + gan_dis_cost_gen \
                       + gen_regularizer_weight * gan_dis_regularization_loss

        gan_total_cost = gan_gen_cost + 8 * gan_gen_FV_cost + \
                         + gen_disentangle_weight * gen_disentangle_cost \
                         + gen_cla_weight * gen_cla_cost \
                         + gen_regularizer_weight * gen_regularization_loss

        val_recon_img = tf.placeholder(tf.float32, [None, self.image_shape[0] * int(math.ceil(self.batch_size ** (.5))),
                 self.image_shape[1] * 3 * int(math.ceil(self.batch_size / math.ceil(self.batch_size ** (.5)))), 3])
        summary_gen_recon_cost = tf.summary.scalar('gen_recon_cost', gen_recon_cost)
        summary_gen_disentangle_cost = tf.summary.scalar('gen_disentangle_cost', gen_disentangle_cost)
        summary_gen_total_cost = tf.summary.scalar('gen_total_cost', gen_total_cost)
        summary_gen_cla_accuracy = tf.summary.scalar('gen_cla_accuracy', gen_cla_accuracy)
        summary_dis_cost = tf.summary.scalar('dis_cost', dis_cost_tf)
        summary_dis_total_cost = tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf)
        summary_dis_prediction_max = tf.summary.scalar('dis_prediction_max', tf.reduce_max([dis_prediction_left[0], dis_prediction_right[0]]))
        summary_dis_prediction_mean = tf.summary.scalar('dis_prediction_mean', (dis_prediction_left[1] + dis_prediction_right[1]) / 2)
        summary_dis_prediction_min = tf.summary.scalar('dis_prediction_min', tf.reduce_min([(dis_prediction_left[2], dis_prediction_right[2])]))
        summary_gan_dis_cost = tf.summary.scalar('gan_dis_cost', gan_dis_cost)
        summary_gan_gen_FV_cost = tf.summary.scalar('gan_gen_FV_cost', gan_gen_FV_cost)
        summary_gan_gen_cost = tf.summary.scalar('gan_gen_cost', gan_gen_cost)
        summary_gan_total_cost = tf.summary.scalar('gan_total_cost', gan_total_cost)
        summary_val_recon_img = tf.summary.image('val_recon_img',val_recon_img)
        summary_merge_scalar = tf.summary.merge(
            [summary_gen_recon_cost, summary_gen_disentangle_cost, summary_gen_total_cost,
             summary_gen_cla_accuracy, summary_dis_cost, summary_dis_total_cost, summary_dis_prediction_max,
             summary_dis_prediction_mean, summary_dis_prediction_min, summary_gan_dis_cost, summary_gan_gen_cost,
             summary_gan_gen_FV_cost,summary_gan_total_cost])
        summary_gen_merge_scalar = tf.summary.merge([summary_gen_recon_cost, summary_gen_disentangle_cost, summary_gen_total_cost,
           summary_gen_cla_accuracy,summary_dis_prediction_max,summary_dis_prediction_mean,
           summary_dis_prediction_min])
        summary_adv_merge_scalar = tf.summary.merge([summary_dis_cost, summary_dis_total_cost, summary_dis_prediction_max
                                                        , summary_dis_prediction_mean, summary_dis_prediction_min])
        summary_gan_gen_merge_scalar = tf.summary.merge([summary_gen_recon_cost, summary_gen_disentangle_cost, summary_gen_cla_accuracy,
                                                     summary_dis_prediction_max,summary_dis_prediction_mean,summary_dis_prediction_min,
                                                     summary_gan_gen_cost, summary_gan_gen_FV_cost, summary_gan_total_cost])
        summary_gan_dis_merge_scalar = tf.summary.merge([summary_gan_dis_cost])
        summary_merge_img = tf.summary.merge([summary_val_recon_img])
        return Y_left, Y_right, image_real_left, image_real_right, gen_recon_cost, gen_disentangle_cost, \
               gen_cla_cost, gen_total_cost, \
               dis_cost_tf, dis_total_cost_tf, image_gen_left, image_gen_right, \
               dis_prediction_left, dis_prediction_right, gen_cla_accuracy, F_I_left, F_V_left, \
               gan_gen_cost, gan_dis_cost, gan_total_cost, val_recon_img, \
               summary_merge_scalar, summary_gen_merge_scalar, summary_adv_merge_scalar, \
               summary_gan_gen_merge_scalar, summary_gan_dis_merge_scalar, summary_merge_img

    def GAN_discriminator(self, image, Y, reuse=False):
        # First convolutional layer - maps one grayscale image to 64 feature maps.
        with tf.name_scope('gan_dis_conv1'):
            h_conv1 = lrelu(
                tf.nn.conv2d(image, self.gan_dis_W1, strides=[1, 1, 1, 1], padding='SAME') + self.gan_dis_b1)
        # First pooling layer - downsamples by 2X.
        with tf.name_scope('gan_dis_pool1'):
            h_pool1 = avg_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 64 feature maps to 128.
        with tf.name_scope('gan_dis_conv2'):
            h_conv2 = tf.nn.conv2d(h_pool1, self.gan_dis_W2, strides=[1, 1, 1, 1], padding='SAME')  # + self.gan_dis_b2
            h_conv2 = lrelu(batchnormalize(h_conv2, 'gan_dis_bn1', train=self.is_training, reuse=reuse,
                                           soft = self.soft_bn, valid = self.train_bn))

        # Second pooling layer.
        with tf.name_scope('gan_dis_pool2'):
            h_pool2 = avg_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('gan_dis_fc1'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
            h_fc1 = lrelu(batchnormalize(tf.matmul(h_pool2_flat, self.gan_dis_W3)  # + self.gan_dis_b3
                                         , 'gan_dis_bn2', soft = self.soft_bn, train=self.is_training,
                                         reuse=reuse, valid = self.train_bn))

        with tf.name_scope('gan_dis_fc2'):
            h_fc2 = tf.matmul(h_fc1, self.gan_dis_W4) + self.gan_dis_b4

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h_fc2))
        return loss

    def gen_disentangle_cost(self, label, logits):
        if self.disentangle_obj_func == 'negative_log':
            return -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=logits))
        if self.disentangle_obj_func == 'even':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.even_label, logits=logits))
        p = tf.nn.softmax(logits)
        if self.disentangle_obj_func == 'one_minus':
            return tf.reduce_mean(self.entropy_calculation(label, 1 - p))
        if self.disentangle_obj_func == 'entropy':
            return -1 * self.entropy_calculation(p, p)
        if self.disentangle_obj_func == 'hybrid':
            return (tf.reduce_mean(self.entropy_calculation(label, 1 - p))
                    -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=logits))) / 2
        return (tf.reduce_mean(self.entropy_calculation(label, 1 - p))
                -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=logits)) -1 * self.entropy_calculation(p, p)) /3

    def entropy_calculation(self, p1, p2):
        p1 = tf.convert_to_tensor(p1)
        p2 = tf.convert_to_tensor(p2)
        precise_p2 = tf.cast(p2, tf.float32) if (
            p2.dtype == tf.float16) else p2
        # labels and logits must be of the same type
        p1 = tf.cast(p1, precise_p2.dtype)
        return tf.reduce_mean(-tf.reduce_sum(p1 * tf.log(p2 + 1e-8), reduction_indices=[1]))

    def encoder(self, image, reuse=False):

        # First convolutional layer - maps one grayscale image to 64 feature maps.
        with tf.name_scope('encoder_conv1'):
            h_conv1 = lrelu(
                tf.nn.conv2d(image, self.encoder_W1, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b1)
        # First pooling layer - downsamples by 2X.
        with tf.name_scope('encoder_pool1'):
            h_pool1 = avg_pool_2x2(h_conv1)


        if self.split_encoder:
            # FI: Second convolutional layer -- maps 64 feature maps to 128.
            with tf.name_scope('encoder_conv2_FI'):
                h_conv2_FI = tf.nn.conv2d(h_pool1, self.encoder_W2_FI, strides=[1, 1, 1, 1],
                                       padding='SAME')  # +self.encoder_b2
                h_conv2_FI = lrelu(batchnormalize(h_conv2_FI, 'en_bn1_FI', soft=self.soft_bn,
                                               train=self.is_training, reuse=reuse, valid=self.train_bn))

            # Second pooling layer.
            with tf.name_scope('encoder_pool2_FI'):
                h_pool2_FI = avg_pool_2x2(h_conv2_FI)

            with tf.name_scope('encoder_fc1_FI'):
                h_pool2_flat_FI = tf.reshape(h_pool2_FI, [-1, 7 * 7 * self.dim_F_I])
                h_fc1_FI = batchnormalize(lrelu(tf.matmul(h_pool2_flat_FI, self.encoder_W3_FI))
                                             , 'fix_scale_en_bn3_FI', soft=self.soft_bn,
                                             train=self.is_training, reuse=reuse, valid=self.train_bn)


            # FV: Second convolutional layer -- maps 64 feature maps to 128.
            with tf.name_scope('encoder_conv2_FV'):
                h_conv2_FV = tf.nn.conv2d(h_pool1, self.encoder_W2_FV, strides=[1, 1, 1, 1],
                                          padding='SAME')  # +self.encoder_b2
                h_conv2_FV = lrelu(batchnormalize(h_conv2_FI, 'en_bn1_FV', soft=self.soft_bn,
                                                  train=self.is_training, reuse=reuse, valid=self.train_bn))

            # Second pooling layer.
            with tf.name_scope('encoder_pool2_FV'):
                h_pool2_FV = avg_pool_2x2(h_conv2_FV)

            with tf.name_scope('encoder_fc1_FV'):
                h_pool2_flat_FV = tf.reshape(h_pool2_FV, [-1, 7 * 7 * (self.dim_W2 - self.dim_F_I)])
                h_fc1_FV = batchnormalize(lrelu(tf.matmul(h_pool2_flat_FV, self.encoder_W3_FV))
                                          , 'fix_scale_en_bn3_FV', soft=self.soft_bn,
                                          train=self.is_training, reuse=reuse, valid=self.train_bn)
            return h_fc1_FI, h_fc1_FV

        with tf.name_scope('encoder_conv2'):
            h_conv2 = tf.nn.conv2d(h_pool1, self.encoder_W2, strides=[1, 1, 1, 1],
                                      padding='SAME')  # +self.encoder_b2
            h_conv2 = lrelu(batchnormalize(h_conv2, 'en_bn1', soft=self.soft_bn,
                                              train=self.is_training, reuse=reuse, valid=self.train_bn))
        # Second pooling layer.
        with tf.name_scope('encoder_pool2'):
            h_pool2 = avg_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.

        with tf.name_scope('encoder_fc1'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
            h_fc1 = lrelu(batchnormalize(tf.matmul(h_pool2_flat, self.encoder_W3)  # + self.encoder_b3
                                         , 'en_bn2', soft = self.soft_bn,
                                         train=self.is_training, reuse=reuse, valid = self.train_bn))

        F_I, F_V = tf.split(h_fc1, [self.dim_F_I, self.dim_W1 - self.dim_F_I], axis=1)
        return batchnormalize(F_I, 'fix_scale_en_bn3', train=self.is_training, reuse=reuse, soft = self.soft_bn),\
               batchnormalize(F_V,'fix_scale_en_bn4',train=self.is_training,reuse=reuse, soft = self.soft_bn)

    def discriminator(self, F_V, reuse=False):
        # 512 to 512
        h1 = lrelu(batchnormalize(tf.matmul(F_V, self.discrim_W1)  # + self.discrim_b1
                                  , 'dis_bn1', soft = self.soft_bn, train=self.is_training, reuse=reuse, valid = self.train_bn))
        # 512 to 10
        h2 = tf.matmul(h1, self.discrim_W2) + self.discrim_b2
        return h2

    def generator(self, F_I, F_V, reuse=False):
        # F_combine 1*128
        F_combine = tf.concat(axis=1, values=[F_I, F_V])
        # h1 1* dim_W2*7*7
        h1 = lrelu(batchnormalize(tf.matmul(F_combine, self.gen_W1)  # + self.gen_b1
                                  , 'gen_bn1', train=self.is_training, soft = self.soft_bn, reuse=reuse, valid = self.train_bn))
        # h1 7*7*dim_W2
        h1 = tf.reshape(h1, [-1, 7, 7, self.dim_W2])
        output_shape_l3 = [self.batch_size, 14, 14, self.dim_W3]
        # h2 14*14*dim_W3
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l3,
                                    strides=[1, 2, 2, 1])  # + self.gen_b2
        h2 = lrelu(batchnormalize(h2, 'gen_bn2', train=self.is_training, reuse=reuse, soft = self.soft_bn, valid = self.train_bn))
        output_shape_l4 = [tf.shape(h2)[0], 28, 28, self.image_shape[-1]]
        # h3 28*28*3
        h3 = tf.add(tf.nn.conv2d_transpose(
            h2, self.gen_W3, output_shape=output_shape_l4, strides=[1, 2, 2, 1]), self.gen_b3)
        return h3

    def classifier(self, F_I, reuse=False):
        return tf.matmul(F_I, self.classifier_W1) + self.classifier_b1


