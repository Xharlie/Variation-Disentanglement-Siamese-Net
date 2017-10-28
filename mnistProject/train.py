import os
import numpy as np
from model import *
from util import *
from load import mnist_with_valid_set
from time import localtime, strftime
import argparse
import result_validation
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

parser.add_argument("--one_minus_D", action="store_true",
                    help="generator's disentanglement loss use one_minus_D loss")

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
        one_minus_D=args.one_minus_D
)

Y_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf, gen_disentangle_cost_tf, gen_cla_cost_tf,\
    gen_total_cost_tf, dis_cost_tf, dis_total_cost_tf, \
    image_gen_left, image_gen_right, dis_prediction_tf_left, dis_prediction_tf_right, gen_cla_accuracy_tf \
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

train_op_gen = tf.train.AdamOptimizer(
    gen_learning_rate, beta1=0.5).minimize(gen_total_cost_tf, var_list=gen_vars + encoder_vars, global_step=global_step)

train_op_discrim = tf.train.AdamOptimizer(
    dis_learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=discrim_vars, global_step=global_step)

iterations = 0
save_path=""

with tf.Session(config=tf.ConfigProto()) as sess:
    try:
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
                    _, summary, gen_recon_cost_val, gen_disentangle_val, gen_cla_cost_val, gen_total_cost_val, \
                            dis_prediction_val_left, dis_prediction_val_right, gen_cla_accuracy_val \
                        = sess.run(
                            [train_op_gen, merged_summary, g_recon_cost_tf,
                             gen_disentangle_cost_tf, gen_cla_cost_tf, gen_total_cost_tf,
                             dis_prediction_tf_left, dis_prediction_tf_right,gen_cla_accuracy_tf],
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
                                Y_tf:Ys,
                                image_tf_real_left: Xs_left,
                                image_tf_real_right: Xs_right
                                })
                    training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    print("=========== updating D ==========")
                    print("iteration:", iterations)
                    print("discriminator loss:", dis_cost_val)
                    print("discriminator total weigthted loss:", dis_total_cost_val)
                    print("discrim left correct prediction's max,mean,min :", dis_prediction_val_left)
                    print("discrim right correct prediction's max,mean,min :", dis_prediction_val_right)


                if np.mod(iterations, drawing_step) == 0:
                    indexTableVal = [[] for i in range(10)]
                    for index in range(len(vaY)):
                        indexTableVal[vaY[index]].append(index)
                    corrRightVal = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal)
                    image_real_left = vaX[0:visualize_dim].reshape([-1, 28, 28, 1]) / 255
                    generated_samples_left = sess.run(
                            image_gen_left,
                            feed_dict={
                                image_tf_real_left: image_real_left,
                                image_tf_real_right: corrRightVal.reshape([-1, 28, 28, 1]) / 255
                                })
                    # since 16 * 8  = batch size * 2
                    save_visualization(image_real_left, generated_samples_left, (16,8), save_path=args.pic_dir_parent+'sample_%04d.jpg' % int(iterations))

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

F_V_classification_conf = {
    "save_path": save_path,
    "trX": trX,
    "trY": trY,
    "vaX": vaX,
    "vaY": vaY,
    "teX": teX,
    "teY": teY,
    "batch_size": batch_size,
    "image_shape": image_shape,
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

# import pdb; pdb.set_trace()

with open(training_logs_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
    json.dump(vars(args), file)
    print("dumped args info to " + training_logs_dir + 'step' + str(iterations) + '_parameter.txt')
    # file.write(json.dump(args))
with open(model_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
    json.dump(vars(args), file)
    print("dumped args info to " + model_dir + 'step' + str(iterations) + '_parameter.txt')



if args.validate_disentanglement:
    tf.reset_default_graph()
    result_validation.validate_F_V_classification_fail(F_V_classification_conf)