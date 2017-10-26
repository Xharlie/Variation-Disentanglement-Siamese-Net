import numpy as np
from F_V_validation_model import *

'''
This function would train a classifier on top of the representation F_V,
make sure it cannot train out the Identity
'''

def validate_F_V_classification_fail(conf):

    F_V_validation_model = F_V_validation(
        batch_size=conf.batch_size,
        image_shape=conf.image_shape,
        dim_y=conf.dim_y,
        dim_W1=conf.dim_W1,
        dim_W2=conf.dim_W2,
        dim_W3=conf.dim_W3,
        dim_F_I=conf.dim_F_I
    )

    train_op = tf.train.AdamOptimizer(
        learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=discrim_vars, global_step=global_step)

    Y_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf, gen_disentangle_cost_tf, gen_total_cost_tf, dis_cost_tf, dis_total_cost_tf, \
    image_gen_left, image_gen_right, dis_prediction_tf_left, dis_prediction_tf_right = VDSN_model.build_model(
        gen_disentangle_weight, gen_regularizer_weight, dis_regularizer_weight)