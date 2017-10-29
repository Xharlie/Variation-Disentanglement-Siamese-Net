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
            dim_W1=128,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=64,
    ):
        super(F_V_validation, self).__init__(
            batch_size=batch_size,
            image_shape=image_shape,
            dim_y=dim_y,
            dim_W1=dim_W1,
            dim_W2=dim_W2,
            dim_W3=dim_W3,
            dim_F_I=dim_F_I)


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

        Y_logits = self.discriminator(F_V)
        Y_prediction_prob = tf.nn.softmax(Y_logits)

        discriminator_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
        regularizer = tf.contrib.layers.l2_regularizer(0.1)

        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)

        dis_cost_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_logits))
        dis_total_cost_tf = dis_cost_tf + dis_regularizer_weight * dis_regularization_loss
        correct_prediction = tf.equal(tf.argmax(Y_prediction_prob, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('dis_cost_tf', dis_cost_tf, collections=['train','test','validation'])
        tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf, collections=['train','test','validation'])
        tf.summary.scalar('accuracy', accuracy, collections=['train','test','validation'])

        return Y, image_real, dis_cost_tf, dis_total_cost_tf, Y_prediction_prob, accuracy



