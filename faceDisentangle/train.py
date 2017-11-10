import os
import numpy as np
from model import *
from util import *
from web_face_load import *
from time import localtime, strftime
import argparse
import glob
# import classification_validation
# import reconst_vali
import json
import math
import copy

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

parser.add_argument("--batch_size", nargs='?', type=int, default=128,
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

parser.add_argument("--gen_series", nargs='?', type=int, default=10,
                    help="how many time the generator can train consecutively")

parser.add_argument("--dis_series", nargs='?', type=int, default=100,
                    help="how many time the dis can train consecutively")

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

parser.add_argument("--pretrain_model", nargs='?', type=str, default='',
                    help="pretrain model")

parser.add_argument("--pretrain_model_wo_lr", nargs='?', type=str, default='',
                    help="pretrain model without learning rate")

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

# >==================  F_V_validation args =======================<

parser.add_argument("--F_V_validation_logs_dir_root", nargs='?', type=str, default='F_V_validation/',
                    help="root dir inside logs_dir_root to save F_V_validation summary")

parser.add_argument("--validate_disentanglement", action="store_true",
                    help="run F_V disentanglement classification task")

parser.add_argument("--validate_classification", action="store_true",
                    help="run F_I classification task")

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
image_shape = [96, 96, 3]
# amount of class label
dim_y=10
dim_11_fltr=32
dim_12_fltr=64
dim_21_fltr=64
dim_22_fltr=64
dim_23_fltr=128
dim_31_fltr=128
dim_32_fltr=96
dim_33_fltr=192
dim_41_fltr=192
dim_42_fltr=128
dim_43_fltr=256
dim_51_fltr=256
dim_52_fltr=160
dim_53_fltr=320
dim_FC=512
dim_F_I=256

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

file_path = "../data/image_sample/*"
directory_list = glob.glob(file_path)

dim_y = len(directory_list)

VDSN_model = VDSN_FACE(
                batch_size=batch_size,
                image_shape=image_shape,
                dim_y=dim_y,
                dim_11_fltr=dim_11_fltr,
                dim_12_fltr=dim_12_fltr,
                dim_21_fltr=dim_21_fltr,
                dim_22_fltr=dim_22_fltr,
                dim_23_fltr=dim_23_fltr,
                dim_31_fltr=dim_31_fltr,
                dim_32_fltr=dim_32_fltr,
                dim_33_fltr=dim_33_fltr,
                dim_41_fltr=dim_41_fltr,
                dim_42_fltr=dim_42_fltr,
                dim_43_fltr=dim_43_fltr,
                dim_51_fltr=dim_51_fltr,
                dim_52_fltr=dim_52_fltr,
                dim_53_fltr=dim_53_fltr,
                dim_FC=dim_FC,
                dim_F_I=dim_F_I,
                simple_discriminator=args.simple_discriminator,
                simple_generator=args.simple_generator,
                simple_classifier=args.simple_classifier,
                disentangle_obj_func='hybrid'
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

dis_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
cla_vars = filter(lambda x: x.name.startswith('cla'), tf.trainable_variables())

# include en_* and encoder_* W and b,
encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())

train_op_gen = tf.train.AdamOptimizer(
    gen_learning_rate, beta1=0.5).minimize(
    gen_total_cost_tf, var_list=gen_vars + encoder_vars + cla_vars, global_step=global_step)

train_op_discrim = tf.train.AdamOptimizer(
    dis_learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=dis_vars, global_step=global_step)

iterations = 0
save_path=""

with tf.Session(config=tf.ConfigProto()) as sess:
    try:
        sess.run(tf.global_variables_initializer())
        # writer for tensorboard summary
        training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
        # test_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
        merged_summary = tf.summary.merge_all()

        if (len(args.pretrain_model)>0):
            # Create a saver. include gen_vars and encoder_vars
            pretrain_saver = tf.train.Saver()
            pretrain_saver.restore(sess, args.pretrain_model)
        elif (len(args.pretrain_model_wo_lr)>0):
            # Create a saver. include gen_vars and encoder_vars
            pretrain_saver = tf.train.Saver(gen_vars + encoder_vars + dis_vars + cla_vars)
            pretrain_saver.restore(sess, args.pretrain_model_wo_lr)

        for epoch in range(n_epochs):
            for i in range(len(directory_list) / batch_size):
                # # pixel value normalized -> from 0 to 1
                # Xs_left = trX[start:end].reshape( [-1, 28, 28, 1])
                # Ys = OneHot(trY[start:end],10)
                #
                # Xs_right = randomPickRight(start, end, trX, trY, indexTable).reshape( [-1, 28, 28, 1])
                left_images, right_images, labels = web_face_load(directory_list, batch_size)
                labels = OneHot(directory_list, len(directory_list))

                left_images = left_images.reshape((-1, 96, 96, 3))
                right_images = right_images.reshape((-1, 96, 96, 3))

                if np.mod( iterations, args.gen_series + args.dis_series ) >= args.dis_series:
                    _, summary, gen_recon_cost_val, gen_disentangle_val, gen_cla_cost_val, gen_total_cost_val, \
                            dis_prediction_val_left, dis_prediction_val_right, gen_cla_accuracy_val \
                        = sess.run(
                            [train_op_gen, merged_summary, g_recon_cost_tf,
                             gen_disentangle_cost_tf, gen_cla_cost_tf, gen_total_cost_tf,
                             dis_prediction_tf_left, dis_prediction_tf_right,gen_cla_accuracy_tf],
                            feed_dict={
                                Y_tf: labels,
                                image_tf_real_left: left_images,
                                image_tf_real_right: right_images
                            })
                    training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    print("=========== updating G ==========")
                    print("iteration:", iterations)
                    print("gen reconstruction loss:", gen_recon_cost_val)
                    print("gen disentanglement loss :", gen_disentangle_val)
                    print("gen id classifier loss :", gen_cla_cost_val)
                    print("total weigthted gen loss :", gen_total_cost_val)
                    print("discrim left correct prediction's max,mean,min:", dis_prediction_val_left)
                    print("discrim right correct prediction's max,mean,min:", dis_prediction_val_right)
                    print("gen id classifier accuracy:", gen_cla_accuracy_val)

                else:
                    _, summary, dis_cost_val, dis_total_cost_val, \
                            dis_prediction_val_left, dis_prediction_val_right \
                        = sess.run(
                            [train_op_discrim, merged_summary, dis_cost_tf, dis_total_cost_tf, \
                             dis_prediction_tf_left, dis_prediction_tf_right],
                            feed_dict={
                                Y_tf: labels,
                                image_tf_real_left: left_images,
                                image_tf_real_right: right_images
                                })
                    training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    print("=========== updating D ==========")
                    print("iteration:", iterations)
                    print("discriminator loss:", dis_cost_val)
                    print("discriminator total weigthted loss:", dis_total_cost_val)
                    print("discrim left correct prediction's max,mean,min :", dis_prediction_val_left)
                    print("discrim right correct prediction's max,mean,min :", dis_prediction_val_right)


                if np.mod(iterations, drawing_step) == 0:
                    image_real_left, image_real_right, _ = web_face_load(directory_list, visualize_dim)
                    generated_samples_left, F_V_matrix, F_I_matrix = sess.run(
                            [image_gen_left, F_V_left_tf, F_I_left_tf],
                            feed_dict={
                                image_tf_real_left: image_real_left,
                                image_tf_real_right: image_real_right
                                })
                    # since 16 * 8  = batch size * 2
                    save_visualization(image_real_left, generated_samples_left,
                                       (int(math.ceil(batch_size ** (.5))),
                                        int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))),
                                       save_path=args.pic_dir_parent + 'sample_%04d.jpg' % int(iterations))
                iterations += 1

        # Save the variables to disk.
        save_path = saver.save(sess, "{}{}_{}_{}_{}.ckpt".format(model_dir,
            gen_regularizer_weight, dis_regularizer_weight, gen_disentangle_weight, time_dir))
        print("Model saved in file: %s" % save_path)

    except KeyboardInterrupt:
        print("Manual interrupt occurred.")
        print('Done training for {} steps'.format(iterations))
        save_path = saver.save(sess, "{}{}_{}_{}_{}.ckpt".format(model_dir,
            gen_regularizer_weight, dis_regularizer_weight, gen_disentangle_weight, time_dir))
        print("Model saved in file: %s" % save_path)

# F_classification_conf = {
#     "save_path": save_path,
#     "batch_size": batch_size,
#     "dim_y": dim_y,
#     "dim_W1": dim_W1,
#     "dim_W2": dim_W2,
#     "dim_W3": dim_W3,
#     "dim_F_I": dim_F_I,
#     "dis_regularizer_weight": args.dis_regularizer_weight,
#     "logs_dir_root": args.logs_dir_root,
#     "F_V_validation_logs_dir_root": args.F_V_validation_logs_dir_root,
#     "F_V_validation_n_epochs": args.F_V_validation_n_epochs,
#     "F_V_validation_learning_rate": args.F_V_validation_learning_rate,
#     "F_V_validation_test_batch_size": args.F_V_validation_test_batch_size,
#     "time_dir": time_dir,
# }
#
# # import pdb; pdb.set_trace()
#
with open(training_logs_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
    json.dump(vars(args), file)
    print("dumped args info to " + training_logs_dir + 'step' + str(iterations) + '_parameter.txt')
    # file.write(json.dump(args))
with open(model_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
    json.dump(vars(args), file)
    print("dumped args info to " + model_dir + 'step' + str(iterations) + '_parameter.txt')



# if args.validate_disentanglement:
#     tf.reset_default_graph()
#     F_V_classification_conf = copy.deepcopy(F_classification_conf)
#     F_V_classification_conf["feature_selection"] = "F_V"
#     classification_validation.validate_F_classification(F_V_classification_conf,trX,trY,vaX,vaY,teX,teY)
#
# if args.validate_classification:
#     tf.reset_default_graph()
#     F_I_classification_conf = copy.deepcopy(F_classification_conf)
#     F_I_classification_conf["feature_selection"] = "F_I"
#     classification_validation.validate_F_classification(F_I_classification_conf,trX,trY,vaX,vaY,teX,teY)
