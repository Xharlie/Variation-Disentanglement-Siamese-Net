import os
import numpy as np
from model import *
from util import *
from cluster import *
from load import mnist_with_valid_set
from time import localtime, strftime
import argparse
import result_validation

parser = argparse.ArgumentParser()
parser.add_argument("--gen_learning_rate", nargs='?', type=float, default=0.0002,
                    help="learning rate")

parser.add_argument("--dis_learning_rate", nargs='?', type=float, default=0.0002,
                    help="learning rate")

parser.add_argument("--batch_size", nargs='?', type=int, default=128,
                    help="batch_size")

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
gen_learning_rate = args.gen_learning_rate
dis_learning_rate = args.dis_learning_rate
batch_size = args.batch_size
image_shape = [28,28,1]
# amount of class label
dim_y=10
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_F_I= 512
time_dir = strftime("%Y-%m-%d-%H-%M-%S", localtime())
gen_disentangle_weight = args.gen_disentangle_weight
gen_regularizer_weight = args.gen_regularizer_weight
dis_regularizer_weight = args.dis_regularizer_weight
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
        dim_F_I=dim_F_I
)

Y_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf, gen_disentangle_cost_tf, gen_total_cost_tf, dis_cost_tf, dis_total_cost_tf, \
    image_gen_left, image_gen_right, dis_prediction_tf_left, dis_prediction_tf_right, F_I_left_tf, F_V_left_tf = VDSN_model.build_model(
    gen_disentangle_weight, gen_regularizer_weight, dis_regularizer_weight)

# saver to save trained model to disk
saver = tf.train.Saver(max_to_keep=10)
# global_step to record steps in total
global_step = tf.Variable(0, trainable=False)

discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

# include en_* and encoder_* W and b,
encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())

train_op_gen = tf.train.AdamOptimizer(
    gen_learning_rate, beta1=0.5).minimize(gen_total_cost_tf, var_list=gen_vars + encoder_vars, global_step=global_step)

train_op_discrim = tf.train.AdamOptimizer(
    dis_learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=discrim_vars, global_step=global_step)

iterations = 0
save_path=""

with tf.Session(config=tf.ConfigProto()) as sess:
    sess.run(tf.global_variables_initializer())
    # writer for tensorboard summary
    training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
    # test_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
    merged_summary = tf.summary.merge_all()

    for epoch in range(n_epochs):
        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]

        indexTable = [[] for i in range(10)]
        for index in range(len(trY)):
            indexTable[trY[index]].append(index)

        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
                ):

            # pixel value normalized -> from 0 to 1
            Xs_left = trX[start:end].reshape( [-1, 28, 28, 1]) / 255.
            Ys = OneHot(trY[start:end],10)

            Xs_right = randomPickRight(start, end, trX, trY, indexTable).reshape( [-1, 28, 28, 1]) / 255.

            if np.mod( iterations, args.gen_odds ) == 0:
                _, summary, gen_recon_cost_val, gen_disentangle_val, gen_total_cost_val, \
                        dis_prediction_val_left, dis_prediction_val_right \
                    = sess.run(
                        [train_op_gen, merged_summary, g_recon_cost_tf, gen_disentangle_cost_tf, gen_total_cost_tf,
                         dis_prediction_tf_left, dis_prediction_tf_right],
                        feed_dict={
                            Y_tf:Ys,
                            image_tf_real_left: Xs_left,
                            image_tf_real_right: Xs_right
                            })
                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                print("=========== updating G ==========")
                print("iteration:", iterations)
                print("gen reconstruction loss:", gen_recon_cost_val)
                print("gen disentanglement loss :", gen_disentangle_val)
                print("total weigthted gen loss :", gen_total_cost_val)
                print("discrim left correct prediction's max,mean,min:", dis_prediction_val_left)
                print("discrim right correct prediction's max,mean,min:", dis_prediction_val_right)

            else:
                _, summary, dis_cost_val, dis_total_cost_val, \
                        dis_prediction_val_left, dis_prediction_val_right \
                    = sess.run(
                        [train_op_discrim, merged_summary, dis_cost_tf, dis_total_cost_tf, \
                         dis_prediction_tf_left, dis_prediction_tf_right],
                        feed_dict={
                            Y_tf:Ys,
                            image_tf_real_left: Xs_left,
                            image_tf_real_right: Xs_right
                            })
                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                print("=========== updating D ==========")
                print("iteration:", iterations)
                print("discriminator loss:", dis_cost_val)
                print("discriminator total weighted loss:", dis_total_cost_val)
                print("discrim left correct prediction's max,mean,min :", dis_prediction_val_left)
                print("discrim right correct prediction's max,mean,min :", dis_prediction_val_right)

            if np.mod(iterations, drawing_step) == 0:
                indexTableVal = [[] for i in range(10)]
                for index in range(len(vaY)):
                    indexTableVal[vaY[index]].append(index)
                corrRightVal = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal)
                image_real_left = vaX[0:visualize_dim].reshape([-1, 28, 28, 1]) / 255
                generated_samples_left, F_I_matrix, F_V_matrix = sess.run(
                        image_gen_left,
                        F_I_left_tf, F_V_left_tf,
                        feed_dict={
                            image_tf_real_left: image_real_left,
                            image_tf_real_right: corrRightVal.reshape([-1, 28, 28, 1]) / 255
                            })
                # since 16 * 8  = batch size * 2
                save_visualization(image_real_left, generated_samples_left, (16,8),
                                   save_path=args.pic_dir_parent+'sample_%04d.jpg' % int(iterations))
                cluster(visualize_dim, vaX, trY[start: end], F_I_matrix, F_V_matrix)
            iterations += 1

    # Save the variables to disk.
    save_path = saver.save(sess, "{}{}_{}_{}_{}.ckpt".format(model_dir,
                                                             gen_regularizer_weight, dis_regularizer_weight,
                                                             gen_disentangle_weight, time_dir
                                                             )
                           )
    print("Model saved in file: %s" % save_path)

F_V_classification_conf = {
    "save_path": save_path,
    "trX": trX,
    "trY": trY,
    "vaX": vaX,
    "vaY": vaY,
    "teX": teX,
    "teY": teY,
    "batch_size":batch_size,
    "image_shape":image_shape,
    "dim_y": dim_y,
    "dim_W1": dim_W1,
    "dim_W2": dim_W2,
    "dim_W3": dim_W3,
    "dim_F_I": dim_F_I,
    "dis_regularizer_weight": args.dis_regularizer_weight,
    "logs_dir_root": args.logs_dir_root,
    "F_V_validation_logs_dir_root": args.F_V_validation_logs_dir_root,
    "F_V_validation_n_epochs": args.F_V_validation_n_epochs,
    "F_V_validation_learning_rate": args.F_V_validation_learning_rate,
    "F_V_validation_test_batch_size": args.F_V_validation_test_batch_size,
    "time_dir":time_dir
}

if args.validate_disentanglement:
    result_validation.validate_F_V_classification_fail(F_V_classification_conf)