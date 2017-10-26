import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
from neural_helper import *
import model

class F_V_validation(model.VDSN):
    def __init__(
            self,
            batch_size=100,
            image_shape=[28, 28, 1],
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=512,
    ):
        super(F_V_validation, self).__init__(batch_size,
            image_shape=[28, 28, 1],
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=512)

        # should be inited by super's __init__
        # self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02), name='discrim_W1')
        # self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        # self.discrim_b1 = bias_variable([self.dim_F_V], name='dis_b1')
        # self.discrim_b2 = bias_variable([self.dim_y], name='dis_b2')
        #
        # self.encoder_W1 = tf.Variable(tf.random_normal([5, 5, image_shape[-1], dim_W3], stddev=0.02), name='encoder_W1')
        # self.encoder_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3, dim_W2], stddev=0.02), name='encoder_W2')
        # self.encoder_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7, dim_W1], stddev=0.02), name='encoder_W3')
        # self.encoder_b1 = bias_variable([dim_W3], name='en_b1')
        # self.encoder_b2 = bias_variable([dim_W2], name='en_b2')
        # self.encoder_b3 = bias_variable([dim_W1], name='en_b3')

    def build_model(self, dis_regularizer_weight=1):
        '''
         Y for class label
        '''
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real = tf.placeholder(tf.float32, [None] + self.image_shape)
        h_fc1 = self.encoder(image_real)

        #  F_V for variance representation
        #  F_I for identity representation
        F_I, F_V = tf.split(h_fc1, num_or_size_splits=2, axis=1)

        Y_logits = self.discriminate(F_V)
        Y_prediction_prob = tf.nn.softmax(Y_logits)

        discriminator_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
        regularizer = tf.contrib.layers.l2_regularizer(0.1)

        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)

        dis_cost_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_logits))
        dis_total_cost_tf = dis_cost_tf + dis_regularizer_weight * dis_regularization_loss


        tf.summary.scalar('dis_cost_tf', dis_cost_tf)
        tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf)

        return Y, image_real, dis_cost_tf, dis_total_cost_tf, Y_prediction_prob

    # def encoder(self, image):
    #     # First convolutional layer - maps one grayscale image to 64 feature maps.
    #     with tf.name_scope('encoder_conv1'):
    #         h_conv1 = lrelu(
    #             tf.nn.conv2d(image, self.encoder_W1, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b1)
    #     # First pooling layer - downsamples by 2X.
    #     with tf.name_scope('encoder_pool1'):
    #         h_pool1 = max_pool_2x2(h_conv1)
    #
    #     # Second convolutional layer -- maps 64 feature maps to 128.
    #     with tf.name_scope('encoder_conv2'):
    #         h_conv2 = tf.nn.conv2d(h_pool1, self.encoder_W2, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b2
    #         h_conv2 = lrelu(batchnormalize(h_conv2))
    #
    #     # Second pooling layer.
    #     with tf.name_scope('encoder_pool2'):
    #         h_pool2 = max_pool_2x2(h_conv2)
    #
    #     # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    #     # is down to 7x7x64 feature maps -- maps this to 1024 features.
    #
    #     with tf.name_scope('encoder_fc1'):
    #         h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
    #         h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.encoder_W3) + self.encoder_b3)
    #
    #     return h_fc1

    def discriminate(self, F_V):
        # 512 to 512
        h1 = lrelu(batchnormalize(tf.matmul(F_V, self.discrim_W1) + self.discrim_b1))
        # 512 to 10
        h2 = lrelu(batchnormalize(tf.matmul(h1, self.discrim_W2) + self.discrim_b2))
        return h2

    # def generator(self, F_I, F_V):
    #     F_combine = tf.concat(axis=1, values=[F_I, F_V])
    #     h1 = tf.nn.relu(batchnormalize(tf.matmul(F_combine, self.gen_W1)))
    #     h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
    #     h2 = tf.reshape(h2, [-1, 7, 7, self.dim_W2])
    #
    #     output_shape_l3 = [self.batch_size, 14, 14, self.dim_W3]
    #     h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
    #     h3 = tf.nn.relu(batchnormalize(h3))
    #     output_shape_l4 = [self.batch_size, 28, 28, self.image_shape[-1]]
    #     h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
    #     return h4


