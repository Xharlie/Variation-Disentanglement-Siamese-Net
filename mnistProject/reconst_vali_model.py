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


    def build_model(self):
        '''
         Y for class label
        '''
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real = tf.placeholder(tf.float32, [None] + self.image_shape)
        h_fc1 = self.encoder(image_real)

        #  F_V for variance representation
        #  F_I for identity representation
        F_I, F_V = tf.split(h_fc1, num_or_size_splits=2, axis=1)

        F_V_validation = tf.zeros_like(F_V)
        h4_validation = self.generator(F_I, F_V_validation)

        image_generated = tf.nn.sigmoid(h4_validation)

        return Y, image_real, image_generated