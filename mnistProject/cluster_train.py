import os
import numpy as np
from model import *
from util import *
from cluster import *
from load import mnist_with_valid_set
from time import localtime, strftime
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--gen_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--dis_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--gen_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--dis_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--gen_decay_rate", nargs='?', type=float, default=0.80,
                    help="generator decay rate")

parser.add_argument("--dis_decay_rate", nargs='?', type=float, default=1.00,
                    help="discriminator decay rate")

parser.add_argument("--batch_size", nargs='?', type=int, default=9999,
                    help="batch_size")

parser.add_argument("--dim_y", nargs='?', type=int, default=10,
                    help="dimension of digit")

parser.add_argument("--dim_W1", nargs='?', type=int, default=128,
                    help="dimension of last encoder layer")

parser.add_argument("--dim_W2", nargs='?', type=int, default=128,
                    help="dimension of second encoder layer ")

parser.add_argument("--dim_W3", nargs='?', type=int, default=64,
                    help="dimension of first encoder layer ")

parser.add_argument("--dim_F_I", nargs='?', type=int, default=64,
                    help="dimension of Identity representation, " +
                         "the dimension of Variation respresentation is dim_W1-dim_F_I")

parser.add_argument("--n_epochs", nargs='?', type=int, default=100,
                    help="number of epochs")

parser.add_argument("--gen_odds", nargs='?', type=int, default=100,
                    help="how many time the generator can train once")

parser.add_argument("--drawing_step", nargs='?', type=int, default=200,
                    help="how many steps to draw a comparision pic")

parser.add_argument("--gen_regularizer_weight", nargs='?', type=float, default=0.01,
                    help="generator regularization weight")

parser.add_argument("--dis_regularizer_weight", nargs='?', type=float, default=0.01,
                    help="discriminator regularization weight")

parser.add_argument("--gen_disentangle_weight", nargs='?', type=float, default=10.0,
                    help="generator disentanglement weight")

parser.add_argument("--gen_cla_weight", nargs='?', type=float, default=5.0,
                    help="generator classifier weight")

parser.add_argument("--logs_dir_root", nargs='?', type=str, default='tensorflow_log/',
                    help="root dir to save training summary")

parser.add_argument("--main_logs_dir_root", nargs='?', type=str, default='main/',
                    help="root dir inside logs_dir_root to save main summary")

parser.add_argument("--model_dir_parent", nargs='?', type=str, default='model_treasury/',
                    help="root dir to save model")

parser.add_argument("--pic_dir_parent", nargs='?', type=str, default='vis/',
                    help="root dir to save pic")

parser.add_argument("--gpu_ind", nargs='?', type=str, default='0',
                    help="which gpu to use")

parser.add_argument("--simple_discriminator", action="store_false",
                    help="simple_discriminator indicate use one fc layer for discriminator")

parser.add_argument("--simple_generator", action="store_false",
                    help="simple_generator indicate use one fc layer for generator")

parser.add_argument("--simple_classifier", action="store_false",
                    help="simple_classifier indicate use one fc layer for classifier")

parser.add_argument("--disentangle_obj_func", nargs='?', type=str, default='negative_log',
                    help="generator's disentanglement loss use which loss, negative_log, one_minus or hybrid")

parser.add_argument("--pretrain_model", nargs='?', type=str, default='',
                    help="pretrain model")

# >==================  F_V_validation args =======================<

parser.add_argument("--F_V_validation_logs_dir_root", nargs='?', type=str, default='F_V_validation/',
                    help="root dir inside logs_dir_root to save F_V_validation summary")

parser.add_argument("--validate_disentanglement", action="store_true",
                    help="run F_V disentanglement classification task")

parser.add_argument("--F_V_validation_n_epochs", nargs='?', type=int, default=100,
                    help="number of epochs for F_V_validation")

parser.add_argument("--F_V_validation_learning_rate", nargs='?', type=float, default=0.0002,
                    help="learning rate for F_V_validation")

parser.add_argument("--F_V_validation_test_batch_size", nargs='?', type=int, default=1000,
                    help="F V validation's test_batch_size")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind

n_epochs = args.n_epochs

batch_size = args.batch_size
image_shape = [28,28,1]
# amount of class label
dim_y=10
dim_W1 = 128
dim_W2 = 128
dim_W3 = 64
dim_F_I= 64
time_dir = strftime("%Y-%m-%d-%H-%M-%S", localtime())
gen_disentangle_weight = args.gen_disentangle_weight
gen_regularizer_weight = args.gen_regularizer_weight
dis_regularizer_weight = args.dis_regularizer_weight
gen_cla_weight = args.gen_cla_weight
# if we don't have these directory, create them
check_create_dir(args.logs_dir_root)
check_create_dir(args.logs_dir_root + args.main_logs_dir_root)
check_create_dir(args.model_dir_parent)
check_create_dir(args.pic_dir_parent)

training_logs_dir = check_create_dir(args.logs_dir_root + args.main_logs_dir_root + time_dir + '/')
model_dir = check_create_dir(args.model_dir_parent + time_dir + '/')

visualize_dim = batch_size
drawing_step = args.drawing_step

# train image validation image, test image, train label, validation label, test label
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

VDSN_model = VDSN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_y=dim_y,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_F_I=dim_F_I,
        simple_discriminator=args.simple_discriminator,
        simple_generator=args.simple_generator,
        simple_classifier=args.simple_classifier,
        disentangle_obj_func=args.disentangle_obj_func
)

Y_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf, gen_disentangle_cost_tf, gen_cla_cost_tf,\
    gen_total_cost_tf, dis_cost_tf, dis_total_cost_tf, \
    image_gen_left, image_gen_right, dis_prediction_tf_left, dis_prediction_tf_right, gen_cla_accuracy_tf, \
    F_I_left_tf, F_V_left_tf,\
    = VDSN_model.build_model(
    gen_disentangle_weight, gen_regularizer_weight, dis_regularizer_weight, gen_cla_weight)

# saver to save trained model to disk
saver = tf.train.Saver(max_to_keep=10)
# global_step to record steps in total
global_step = tf.Variable(0, trainable=False)
gen_learning_rate = tf.train.exponential_decay(args.gen_start_learning_rate, global_step,
                                               args.gen_decay_step, args.gen_decay_rate, staircase=True)
dis_learning_rate = tf.train.exponential_decay(args.dis_start_learning_rate, global_step,
                                               args.dis_decay_step, args.dis_decay_rate, staircase=True)

discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

# include en_* and encoder_* W and b,
encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())

iterations = 0
save_path=""

with tf.Session(config=tf.ConfigProto()) as sess:

    sess.run(tf.global_variables_initializer())
    pretrain_saver = tf.train.Saver(encoder_vars)
    pretrain_saver.restore(sess, args.pretrain_model)
    for start, end in zip(
            range(0, len(teX), batch_size),
            range(batch_size, len(teY), batch_size)
            ):

        indexTableVal = [[] for i in range(10)]
        for index in range(len(teY)):
            indexTableVal[teY[index]].append(index)
        corrRightVal,_ = randomPickRight(0, visualize_dim, teX, teY, indexTableVal)
        image_real_left = teX[0:visualize_dim].reshape([-1, 28, 28, 1]) / 255
        generated_samples_left, F_V_matrix, F_I_matrix = sess.run(
                [image_gen_left, F_V_left_tf, F_I_left_tf],
                feed_dict={
                    image_tf_real_left: image_real_left,
                    image_tf_real_right: corrRightVal.reshape([-1, 28, 28, 1]) / 255
                    })
        cluster(image_real_left, teY[0: visualize_dim], F_I_matrix, F_V_matrix, iterations=iterations)
        iterations += 1

