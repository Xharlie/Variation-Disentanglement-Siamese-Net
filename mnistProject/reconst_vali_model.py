import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
from neural_helper import *
import model

class reconst_validation_model(model.VDSN):
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
        super(reconst_validation_model, self).__init__(
            batch_size=batch_size,
            image_shape=image_shape,
            dim_y=dim_y,
            dim_W1=dim_W1,
            dim_W2=dim_W2,
            dim_W3=dim_W3,
            dim_F_I=dim_F_I)
    '''  
     F_I_0 for F_I and 00,
     F_I_C for F_I and F center 
     F_I_F_V for F_I and F_V from same digit different images,  
     F_I_F_D_F_V for F_I with F_V from different digit
    '''

    def build_model(self, feature_selection):

        Y = tf.placeholder(tf.float32, [None, self.dim_y])
        image_real_left = tf.placeholder(tf.float32, [None] + self.image_shape)
        image_real_right = tf.placeholder(tf.float32, [None] + self.image_shape)
        center_representation = tf.placeholder(tf.float32, [None] + [self.dim_F_V])

        #  image_F_I for image source of identity representation
        #  image_F_V for image source of variance representation
        #  image_generated for generated image;
        #  we r gonna show them side by side

        if feature_selection=="F_I_F_V" or feature_selection=="F_I_F_D_F_V":
            h_fc1_left = self.encoder(image_real_left)
            h_fc1_right = self.encoder(image_real_right)
            F_I_left, F_V_left = tf.split(h_fc1_left, num_or_size_splits=2, axis=1)
            F_I_right, F_V_right = tf.split(h_fc1_right, num_or_size_splits=2, axis=1)
            h4_validation = self.generator(F_I_left, F_V_right)
            image_F_I = image_real_left
            image_F_V = image_real_right
            image_generated = tf.nn.sigmoid(h4_validation)
        else:
            h_fc1_left = self.encoder(image_real_left)
            F_I_left, F_V_left = tf.split(h_fc1_left, num_or_size_splits=2, axis=1)
            h4_validation = self.generator(F_I_left, center_representation)
            image_F_I = image_real_left
            image_F_V = tf.nn.sigmoid(self.generator(
                tf.zeros_like(F_I_left), center_representation))
            image_generated = tf.nn.sigmoid(h4_validation)
        return Y, center_representation, image_real_left, \
               image_real_right, image_F_I, image_F_V, image_generated

    def build_class_center(self):

        Y = tf.placeholder(tf.float32, [None])
        image_real = tf.placeholder(tf.float32, [None] + self.image_shape)
        h_fc1 = self.encoder(image_real)
        F_I, F_V = tf.split(h_fc1, num_or_size_splits=2, axis=1)
        return Y, image_real, F_I, F_V