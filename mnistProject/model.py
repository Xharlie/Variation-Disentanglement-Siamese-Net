#-*- coding: utf-8 -*-
import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
from neural_helper import *

class VDSN(object):
    def __init__(
            self,
            batch_size=100,
            image_shape=[28,28,1],
            dim_y=10,
            dim_W1=128,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=64,
            simple_discriminator=True,
            simple_generator=True,
            simple_classifier=True,
            disentangle_obj_func = 'hybrid'
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_F_I = dim_F_I
        self.dim_F_V = dim_W1 - dim_F_I
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.simple_discriminator = simple_discriminator
        self.simple_generator = simple_generator
        self.simple_classifier = simple_classifier
        # disentangle_obj_func = negative_log (-logD(x)), one_minus(log(1-D(x))) or hybrid
        self.disentangle_obj_func=disentangle_obj_func

        if not self.simple_generator:
            self.gen_W1 = tf.Variable(tf.random_normal([dim_W1, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1, dim_W2*7*7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,image_shape[-1],dim_W3], stddev=0.02), name='gen_W4')

        if not self.simple_discriminator:
            self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02), name='discrim_W1')
            self.discrim_b1 = bias_variable([self.dim_F_V], name='dis_b1')
        self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        self.discrim_b2 = bias_variable([self.dim_y], name='dis_b2')

        self.encoder_W1 = tf.Variable(tf.random_normal([5, 5, image_shape[-1] , dim_W3], stddev=0.02),name='encoder_W1')
        self.encoder_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3 , dim_W2], stddev=0.02), name='encoder_W2')
        self.encoder_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7 , dim_W1], stddev=0.02),name='encoder_W3')
        self.encoder_b1 = bias_variable([dim_W3],name='en_b1')
        self.encoder_b2 = bias_variable([dim_W2],name='en_b2')
        self.encoder_b3 = bias_variable([dim_W1],name='en_b3')

        if not self.simple_classifier:
            self.classifier_W1 = tf.Variable(tf.random_normal([self.dim_F_I, self.dim_F_I], stddev=0.02), name='classif_W1')
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
        F_I_left, F_V_left = tf.split(h_fc1_left, num_or_size_splits=2, axis = 1)
        F_I_right, F_V_right = tf.split(h_fc1_right, num_or_size_splits=2, axis = 1)

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

        dis_prediction_left = [tf.reduce_max(Y_dis_result_left), tf.reduce_mean(Y_dis_result_left), tf.reduce_min(Y_dis_result_left)];
        dis_prediction_right = [tf.reduce_max(Y_dis_result_right), tf.reduce_mean(Y_dis_result_right), tf.reduce_min(Y_dis_result_right)];

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
            regularizer, weights_list= gen_vars + encoder_vars + classifier_vars)
        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)
        
        gen_recon_cost_left = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size
        gen_recon_cost_right = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size

        gen_disentangle_cost_left = self.gen_disentangle_cost(Y,Y_dis_logits_left)
        gen_disentangle_cost_right = self.gen_disentangle_cost(Y,Y_dis_logits_right)

        gen_cla_cost_left=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_cla_logits_left))
        gen_cla_cost_right=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_cla_logits_right))

        dis_loss_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_dis_logits_left))
        dis_loss_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_dis_logits_right))

        gen_recon_cost = (gen_recon_cost_left + gen_recon_cost_right) / 2
        gen_disentangle_cost = (gen_disentangle_cost_left + gen_disentangle_cost_right) / 2
        gen_cla_cost = (gen_cla_cost_left + gen_cla_cost_right) / 2
        gen_total_cost = gen_recon_cost \
                         + gen_disentangle_weight * gen_disentangle_cost \
                         + gen_cla_weight * gen_cla_cost    \
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
        tf.summary.scalar('dis_prediction_mean', (dis_prediction_left[1] + dis_prediction_right[1])/2 )
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
        with tf.name_scope('encoder_conv1'):
            h_conv1 = lrelu(tf.nn.conv2d(image, self.encoder_W1, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b1)
        # First pooling layer - downsamples by 2X.
        with tf.name_scope('encoder_pool1'):
            h_pool1 = avg_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 64 feature maps to 128.
        with tf.name_scope('encoder_conv2'):
            h_conv2 = tf.nn.conv2d(h_pool1, self.encoder_W2, strides=[1, 1, 1, 1], padding='SAME')+self.encoder_b2
            h_conv2 = lrelu(batchnormalize(h_conv2))

        # Second pooling layer.
        with tf.name_scope('encoder_pool2'):
            h_pool2 = avg_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.

        with tf.name_scope('encoder_fc1'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
            h_fc1 = lrelu(batchnormalize(tf.matmul(h_pool2_flat, self.encoder_W3) + self.encoder_b3))

        return h_fc1

    def discriminator(self, F_V):
        # 512 to 512
        h1 =F_V
        if not self.simple_discriminator:
            h1 = lrelu(batchnormalize(tf.matmul(F_V, self.discrim_W1) + self.discrim_b1))
            # 512 to 10
        h2 = tf.matmul(h1, self.discrim_W2) + self.discrim_b2
        return h2

    def generator(self, F_I,F_V):
	F_I = batchnormalize(F_I)
	F_V = batchnormalize(F_V)
        # F_combine = tf.concat(axis=1, values=[F_I,F_V])
        F_combine = tf.add(F_I, F_V)
	F_combine = tf.concat(axis=1, values=(F_combine, F_combine))
	h1 = F_combine
        if not self.simple_generator:
            h1 = lrelu(batchnormalize(tf.matmul(F_combine, self.gen_W1)))
        h2 = lrelu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [-1,7,7,self.dim_W2])

        output_shape_l3 = [tf.shape(h2)[0],14,14,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = lrelu(batchnormalize(h3))
        output_shape_l4 = [tf.shape(h3)[0],28,28,self.image_shape[-1]]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4

    def classifier(self, F_I):
        h1 = F_I
        # if not self.simple_discriminator:
        #    h1 = lrelu(batchnormalize(tf.matmul(F_I, self.classifier_W1) + self.classifier_b1))
            # 512 to 10
        h2 = tf.matmul(h1, self.classifier_W2) + self.classifier_b2
        return h2
