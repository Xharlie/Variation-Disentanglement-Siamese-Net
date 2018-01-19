import sys
sys.path.append("../")
from neural_helper import *
import model

class F_validation(model.VDSN):
    def __init__(
            self,
            batch_size=100,
            image_shape=[28, 28, 1],
            dim_y=10,
            dim_W1=128,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=64,
            split_encoder = True
    ):
        super(F_validation, self).__init__(
            batch_size=batch_size,
            image_shape=image_shape,
            dim_y=dim_y,
            dim_W1=dim_W1,
            dim_W2=dim_W2,
            dim_W3=dim_W3,
            dim_F_I=dim_F_I,
            split_encoder = split_encoder)


    def build_model(self, feature_selection='F_V', dis_regularizer_weight=1):
        '''
         Y for class label
        '''
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real = tf.placeholder(tf.float32, [None] + self.image_shape)

        F_I, F_V = self.encoder(image_real,reuse=False)

        F_target = F_V
        if feature_selection == 'F_I':
            F_target = F_I


        Y_logits = self.discriminator(F_target,reuse=False)
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
        #----------------------------- generated classification -----------------------------------

        image_real_support = tf.placeholder(tf.float32, [None] + self.image_shape)
        F_I_support, F_V_support = self.encoder(image_real_support, reuse=True)
        h3_support = self.generator(F_I, F_V_support, reuse=False)
        h3_real = self.generator(F_I_support, F_V, reuse=True)
        image_gen_real = tf.nn.sigmoid(h3_real)
        image_gen_support = tf.nn.sigmoid(h3_support)
        shape = self.image_shape[0] * self.image_shape[1] * self.batch_size
        gen_recon_cost_real = tf.nn.l2_loss(image_real - image_gen_real) / shape
        gen_recon_cost_suport = tf.nn.l2_loss(image_real_support - image_gen_support) / shape
        gen_recon_cost = (gen_recon_cost_real + gen_recon_cost_suport) / 2

        F_I_gen_support, F_V_gen_support = self.encoder(image_gen_support, reuse=True)
        Y_logits_gen = self.discriminator(F_I_gen_support, reuse=True)
        Y_prediction_prob_gen = tf.nn.softmax(Y_logits_gen)
        correct_prediction_gen = tf.equal(tf.argmax(Y_prediction_prob_gen, 1), tf.argmax(Y, 1))
        accuracy_gen = tf.reduce_mean(tf.cast(correct_prediction_gen, tf.float32))

        return Y, image_real, image_real_support, dis_cost_tf, dis_total_cost_tf, Y_prediction_prob, accuracy, gen_recon_cost, accuracy_gen



