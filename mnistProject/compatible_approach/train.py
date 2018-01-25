from model import *
from util import *
from neural_helper import *
from cluster import *
from load import mnist_with_valid_set
from time import localtime, strftime
import argparse
import classification_validation
import json
import math
import copy

parser = argparse.ArgumentParser()

parser.add_argument("--dis_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--dis_decay_rate", nargs='?', type=float, default=1.00,
                    help="discriminator decay rate")

parser.add_argument("--dis_start_learning_rate", nargs='?', type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--dis_regularizer_weight", nargs='?', type=float, default=0.01,
                    help="discriminator regularization weight")

parser.add_argument("--gen_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--gan_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--gen_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--gen_decay_rate", nargs='?', type=float, default=0.80,
                    help="generator decay rate")

parser.add_argument("--gan_decay_rate", nargs='?', type=float, default=1.00,
                    help="GAN discriminator decay rate")

parser.add_argument("--gan_decay_step", nargs='?', type=float, default=10000,
                    help="GAN decay step")

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

parser.add_argument("--recon_epoch", nargs='?', type=int, default=40,
                    help="how many time the generator with reconstruction task can train consecutively")

parser.add_argument("--gan_epoch", nargs='?', type=int, default=60,
                    help="how many time the generator with gan task can train consecutively")

parser.add_argument("--drawing_epoch", nargs='?', type=int, default=1,
                    help="how many epochs to draw a comparision pic")

parser.add_argument("--save_epoch", nargs='?', type=int, default=50,
                    help="how many epoch to save")

parser.add_argument("--gen_regularizer_weight", nargs='?', type=float, default=0.01,
                    help="generator regularization weight")

parser.add_argument("--gen_metric_loss_weight", nargs='?', type=float, default=10.0,
                    help="metric loss")

parser.add_argument("--gen_cla_weight", nargs='?', type=float, default=5.0,
                    help="generator classifier weight")

parser.add_argument("--gan_gen_weight", nargs='?', type=float, default=1.0,
                    help="gan_gen_weight weight")

parser.add_argument("--F_IV_recon_weight", nargs='?', type=float, default=1.0,
                    help="F_IV_recon_weight weight")

parser.add_argument("--F_I_gen_recon_weight", nargs='?', type=float, default=1.0,
                    help="F_I_gen_recon_weight weight")

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

parser.add_argument("--recon_img_not_tensorboard", action="store_true",
                    help="save validation reconstruction image locally not tensorboard")

parser.add_argument("--gpu_ind", nargs='?', type=str, default='0',
                    help="which gpu to use")

parser.add_argument("--debug", action="store_true",
                    help="debug_mode")

parser.add_argument("--train_bn", action="store_true",
                    help="train bn's gamma and beta")

parser.add_argument("--soft_bn", action="store_true",
                    help="use moving in training")

parser.add_argument("--metric_obj_func", nargs='?', type=str, default='central',
                    help="generator's disentanglement loss use which loss, negative_log, one_minus, hybrid or complex")

parser.add_argument("--summary_update_step", nargs='?', type=int, default=10,
                    help="every how many steps we should update summary")

parser.add_argument("--avg_momentum", nargs='?', type=float, default=0.05,
                    help="momentum when updating average value")

parser.add_argument("--metric_norm", nargs='?', type=int, default=2,
                    help="metric loss norm")

parser.add_argument("--critics_iters", nargs='?', type=int, default=5,
                    help="critics iterations")

parser.add_argument("--clustering_tsb", action="store_true",
                    help="save clustering to tensorboard")

parser.add_argument("--not_split_encoder", action="store_true",
                        help="use moving in training")

parser.add_argument("--F_I_batch_norm", action="store_true",
                        help="use batch_norm in F_I")

parser.add_argument("--F_V_limit_variance", action="store_true",
                        help="limit variance for in F_V")

parser.add_argument("--F_multiply", action="store_true",
                        help="limit variance for in F_V")


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

parser.add_argument("--F_V_validation_test_batch_size", nargs='?', type=int, default=128,
                    help="F V validation's test_batch_size")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind

n_epochs = args.recon_epoch + args.gan_epoch

batch_size = args.batch_size
image_shape = [28,28,1]
# amount of class label
dim_y=10
dim_W1 = 128
dim_W2 = 128
dim_W3 = 64
dim_F_I= 64
time_dir = strftime("%Y-%m-%d-%H-%M-%S", localtime())
print "dir:",time_dir
gen_metric_loss_weight = args.gen_metric_loss_weight
gen_regularizer_weight = args.gen_regularizer_weight
gen_cla_weight = args.gen_cla_weight
# if we don't have these directory, create them
check_create_dir(args.logs_dir_root)
check_create_dir(args.logs_dir_root + args.main_logs_dir_root)
check_create_dir(args.model_dir_parent)
check_create_dir(args.pic_dir_parent)
check_create_dir(args.pic_dir_parent+time_dir)

training_logs_dir = check_create_dir(args.logs_dir_root + args.main_logs_dir_root + time_dir + '/')
model_dir = check_create_dir(args.model_dir_parent + time_dir + '/')

visualize_dim = batch_size


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
        metric_obj_func=args.metric_obj_func,
        train_bn= args.train_bn,
        soft_bn= args.soft_bn,
        split_encoder = (not args.not_split_encoder),
        metric_norm = args.metric_norm,
        F_I_batch_norm = args.F_I_batch_norm,
        F_V_limit_variance = args.F_V_limit_variance,
        F_multiply = args.F_multiply
)

Y_left, Y_right, Y_diff, image_real_left, image_real_right, image_real_diff, gen_recon_cost, gen_metric_loss, \
   gen_cla_cost, gen_total_cost, image_gen_left, image_gen_right, gen_cla_accuracy, F_I_left, F_V_left, \
   gan_gen_cost, F_IV_recon_cost, gan_dis_total_cost, gan_gen_total_cost, val_recon_img, F_I_cluster_img, F_V_cluster_img, \
   F_I_center_left, F_I_center_right, F_I_center_diff, F_I_right, F_V_right, F_I_right_IV_right_left_gen_out, \
   F_I_diff_IV_diff_left_gen_out,F_I_diff_V_left_gen_out, F_IV_out_diff_stitch_img, F_IV_out_same_stitch_img, \
   F_V_out_diff_stitch_img, summary_merge_scalar, summary_gen_merge_scalar,summary_gan_gen_merge_scalar, \
   summary_gan_dis_merge_scalar, summary_merge_recon_img, summary_merge_cluster_img, summary_merge_stitch_img \
    = VDSN_model.build_model(
    gen_metric_loss_weight, gen_regularizer_weight, gen_cla_weight, args.gan_gen_weight, args.F_IV_recon_weight, args.F_I_gen_recon_weight)

global_step = tf.Variable(0, trainable=False)
# saver to save trained model to disk
saver = tf.train.Saver(max_to_keep=10)
# global_step to record steps in total
gen_learning_rate = tf.train.exponential_decay(args.gen_start_learning_rate, global_step,
                                               args.gen_decay_step, args.gen_decay_rate, staircase=True)
gan_learning_rate = tf.train.exponential_decay(args.gan_start_learning_rate, global_step,
                                               args.gan_decay_step, args.gan_decay_rate, staircase=True)

if args.train_bn:
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    cla_vars = filter(lambda x: x.name.startswith('cla'), tf.trainable_variables())
    gan_dis_vars = filter(lambda x: x.name.startswith('gan'), tf.trainable_variables())
    # include en_* and encoder_* W and b,
    encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())
    dup_encoder_vars = filter(lambda x: x.name.startswith('dup_en'), tf.trainable_variables())
    filter_vars = filter(lambda x: x.name.startswith('fil'), tf.trainable_variables())

else:
    gen_vars = filter(lambda x: x.name.startswith('gen') and 'bn' not in x.name, tf.trainable_variables())
    cla_vars = filter(lambda x: x.name.startswith('cla') and 'bn' not in x.name, tf.trainable_variables())
    gan_dis_vars = filter(lambda x: x.name.startswith('gan') and 'bn' not in x.name, tf.trainable_variables())
    encoder_vars = filter(lambda x: x.name.startswith('en') and 'bn' not in x.name, tf.trainable_variables())
    dup_encoder_vars = filter(lambda x: x.name.startswith('dup_en'), tf.trainable_variables())
    filter_vars = filter(lambda x: x.name.startswith('fil') and 'bn' not in x.name, tf.trainable_variables())

fix_vars = filter_vars = filter(lambda x: x.name.startswith('fix'), tf.trainable_variables())
dup_fix_vars = filter_vars = filter(lambda x: x.name.startswith('dup_fix'), tf.trainable_variables())

with tf.control_dependencies(tf.get_collection(GEN_BATCH_NORM_OPS)):
    train_op_gen = tf.train.AdamOptimizer(
        gen_learning_rate, beta1=0.5).minimize(
        gen_total_cost, var_list=gen_vars+encoder_vars+cla_vars, global_step=global_step)

with tf.control_dependencies(tf.get_collection(GEN_BATCH_NORM_OPS)):
    train_op_gan_gen = tf.train.AdamOptimizer(
        gan_learning_rate, beta1=0.5).minimize(
        gan_gen_total_cost, var_list=gen_vars+encoder_vars+cla_vars+filter_vars, global_step=global_step)

with tf.control_dependencies(tf.get_collection(DIS_BATCH_NORM_OPS)):
    # not update global step since we considered gan as a whole step
    train_op_gan_discrim = tf.train.AdamOptimizer(
        gan_learning_rate, beta1=0.5).minimize(
        gan_dis_total_cost, var_list=gan_dis_vars)

iterations = 0
save_path=""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
center_init = True;

def resetIndex():
    start = 0
    end = args.batch_size
    return start, end

def get_center_batch(centers, labels, dim):
    centers_in_batch = np.zeros((labels.shape[0], dim), dtype=np.float32)
    for i in range(labels.shape[0]):
        centers_in_batch[i,:] = centers[labels[i]]
    return centers_in_batch

def update_centers(F_I, Ys_index, dim):
    batch_avg = np.zeros((dim_y, dim), dtype=np.float32)
    batch_count = np.ones((dim_y, 1), dtype=np.int)
    batch_updated = np.zeros((dim_y, 1), dtype=np.int)
    for i in range(Ys_index.shape[0]):
        batch_avg[Ys_index[i]] += F_I[i,:]
        if (batch_updated[Ys_index[i]]) == 0:
            batch_updated[Ys_index[i]] = 1
        else:
            batch_count[Ys_index[i]] += 1
    global center_init
    if center_init:
        VDSN_model.runing_avg = batch_avg / batch_count
        center_init = False
    else:
        VDSN_model.runing_avg +=  args.avg_momentum * (batch_avg / batch_count - VDSN_model.runing_avg)

image_real_left_agg = None
F_V_matrix_agg = None
F_I_matrix_agg = None

sess = tf.Session(config=config)
with sess.as_default():
    try:
        sess.run(tf.global_variables_initializer())
        # writer for tensorboard summary
        training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
        # img_writer = tf.summary.FileWriter(training_logs_dir)


        if (len(args.pretrain_model)>0):
            # Create a saver. include gen_vars and encoder_vars
            saver.restore(sess, args.pretrain_model)
        elif (len(args.pretrain_model_wo_lr)>0):
            # Create a saver. include gen_vars and encoder_vars
            pretrain_saver = tf.train.Saver(gen_vars + encoder_vars + cla_vars)
            pretrain_saver.restore(sess, args.pretrain_model_wo_lr)

        index_left = np.arange(len(trY))

        indexTable = [[] for i in range(dim_y)]
        for index in range(len(trY)):
            indexTable[trY[index]].append(index)
        indexTableVal = [[] for i in range(10)]
        for index in range(len(vaY)):
            indexTableVal[vaY[index]].append(index)
        start, end = resetIndex()
        np.random.shuffle(index_left)

        epoch = 0

        def add_tensorboard(originData, target, feature_I, feature_V, pretrain_model_time_dir, iterations):
            print "cluster_thread: begin"
            F_I_cluster_np, F_V_cluster_np = cluster(originData, target, feature_I, feature_V, pretrain_model_time_dir, iterations)
            summary = sess.run(summary_merge_cluster_img, feed_dict={
                F_I_cluster_img: np.expand_dims(F_I_cluster_np, axis=0),
                F_V_cluster_img: np.expand_dims(F_V_cluster_np, axis=0)})
            training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
            print "cluster_thread: added img to summary"

        while epoch < n_epochs:

            print "epoch:", epoch
            while end <= len(trY):
                sess.run(tf.get_collection('update_dup'))
                # print "dup_encoder_vars", sess.run(dup_encoder_vars[2])
                # print "encoder_vars", sess.run(encoder_vars[2])
                # print "dup_fix_vars", sess.run(dup_fix_vars[2])
                # print "fix_vars", sess.run(fix_vars[2])

                Xs_left = trX[index_left[start:end]].reshape([-1, 28, 28, 1]) / 255.
                Ys_left = OneHot(trY[index_left[start:end]], 10)
                Y_left_index = trY[index_left[start:end]]
                Xs_right, _ = randomPickRight(start, end, trX,
                                                     trY[index_left[start:end]], indexTable)
                Xs_right = Xs_right.reshape([-1, 28, 28, 1]) / 255.
                Ys_right = Ys_left
                Y_right_index = Y_left_index
                Xs_diff, Y_diff_index = randomPickRight(start, end, trX,
                                                     trY[index_left[start:end]], indexTable,
                                                     feature="F_I_F_D_F_V")
                Ys_diff = OneHot(Y_diff_index, 10)
                Xs_diff = Xs_diff.reshape([-1, 28, 28, 1]) / 255.


                if epoch < args.recon_epoch:

                    _, summary, gen_recon_cost_val, gen_metric_loss_val, gen_cla_cost_val, gen_total_cost_val, \
                    gen_cla_accuracy_val, F_I_left_val, F_I_right_val \
                        = sess.run(
                        [train_op_gen, summary_gen_merge_scalar, gen_recon_cost,
                         gen_metric_loss, gen_cla_cost, gen_total_cost, gen_cla_accuracy, F_I_left, F_I_right],
                        feed_dict={
                            Y_left: Ys_left,
                            Y_right: Ys_right,
                            Y_diff: Ys_diff,
                            image_real_left: Xs_left,
                            image_real_right: Xs_right,
                            image_real_diff: Xs_diff,
                            F_I_center_left: get_center_batch(VDSN_model.runing_avg, Y_left_index, dim_F_I),
                            F_I_center_right: get_center_batch(VDSN_model.runing_avg, Y_right_index, dim_F_I),
                            F_I_center_diff: get_center_batch(VDSN_model.runing_avg, Y_diff_index, dim_F_I)
                        })
                    update_centers(np.concatenate((F_I_left_val, F_I_right_val),axis=0),
                                   np.concatenate((Y_left_index, Y_right_index),axis=0), dim_F_I)
                    # print "global_step", tf.train.global_step(sess, global_step)
                    if np.mod(tf.train.global_step(sess, global_step), args.summary_update_step) == 0:
                        training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    print("=========== updating G ==========")
                    print("iteration:", iterations)
                    print("gen reconstruction loss:", gen_recon_cost_val)
                    print("gen metric loss :", gen_metric_loss_val)
                    print("gen id classifier loss :", gen_cla_cost_val)
                    print("total weigthted gen loss :", gen_total_cost_val)
                    print("gen id classifier accuracy:", gen_cla_accuracy_val)
                    if args.debug:
                        bn3_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn3/beta'), tf.all_variables())[0]
                        bn3_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn3/gamma'), tf.all_variables())[0]
                        bn4_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn4/beta'), tf.all_variables())[0]
                        bn4_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn4/gamma'), tf.all_variables())[0]
                        print("fix_scale_en_bn3 beta/gamma:" + str(sess.run(bn3_beta)) + "/"+str(sess.run(bn3_gamma)))
                        print("fix_scale_en_bn4 beta/gamma:" + str(sess.run(bn4_beta)) + "/"+str(sess.run(bn4_gamma)))

                else:
                    # start to train gan, D first
                    modulus = np.mod(iterations, args.critics_iters)
                    if modulus != 0:
                        _, summary, gan_dis_total_cost_val , F_I_left_val, F_I_right_val \
                            = sess.run(
                            [train_op_gan_discrim, summary_gan_dis_merge_scalar, gan_dis_total_cost, F_I_left, F_I_right],
                            feed_dict={
                                Y_left: Ys_left,
                                Y_right: Ys_right,
                                Y_diff: Ys_diff,
                                image_real_left: Xs_left,
                                image_real_right: Xs_right,
                                image_real_diff: Xs_diff,
                                F_I_center_left: get_center_batch(VDSN_model.runing_avg, Y_left_index, dim_F_I),
                                F_I_center_right: get_center_batch(VDSN_model.runing_avg, Y_right_index, dim_F_I),
                                F_I_center_diff: get_center_batch(VDSN_model.runing_avg, Y_diff_index, dim_F_I)
                            }
                        )
                        print("=========== updating gan D ==========")
                        print("iteration:", iterations)
                        print("gan_dis_cost:", gan_dis_total_cost_val)
                        if np.mod(tf.train.global_step(sess, global_step), args.summary_update_step) == 0:
                            training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                        # moving_mean = filter(lambda x: x.name.startswith('gan_dis_bn1/moving_mean'), tf.all_variables())[0]
                        # print(moving_mean.name + ":" + str(sess.run(moving_mean)))
                    else:
                        # train G
                        _, summary, gan_gen_cost_val, F_IV_recon_cost_val, gen_metric_loss_val, gen_cla_cost_val, gan_total_cost_val,\
                        gen_cla_accuracy_val, F_I_left_val, F_I_right_val \
                            = sess.run(
                            [train_op_gan_gen, summary_gan_gen_merge_scalar, gan_gen_cost, F_IV_recon_cost,
                             gen_metric_loss, gen_cla_cost, gan_gen_total_cost, gen_cla_accuracy, F_I_left, F_I_right],
                            feed_dict={
                                Y_left: Ys_left,
                                Y_right: Ys_right,
                                Y_diff: Ys_diff,
                                image_real_left: Xs_left,
                                image_real_right: Xs_right,
                                image_real_diff: Xs_diff,
                                F_I_center_left: get_center_batch(VDSN_model.runing_avg, Y_left_index, dim_F_I),
                                F_I_center_right: get_center_batch(VDSN_model.runing_avg, Y_right_index, dim_F_I),
                                F_I_center_diff: get_center_batch(VDSN_model.runing_avg, Y_diff_index, dim_F_I)
                            })
                        update_centers(np.concatenate((F_I_left_val, F_I_right_val), axis=0),
                                       np.concatenate((Y_left_index, Y_right_index), axis=0), dim_F_I)
                        print("=========== updating gan G ==========")
                        print("iteration:", iterations)
                        print("gan gen loss:", gan_gen_cost_val)
                        print("gen metric loss :", gen_metric_loss_val)
                        print("gen id classifier loss :", gen_cla_cost_val)
                        print("total weigthted gan loss :", gan_total_cost_val)
                        print("gen id classifier accuracy:", gen_cla_accuracy_val)
                        if np.mod(tf.train.global_step(sess, global_step), args.summary_update_step) == 0:
                            training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                        if args.debug:
                            bn3_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn3/beta'), tf.all_variables())[0]
                            bn3_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn3/gamma'), tf.all_variables())[0]
                            bn4_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn4/beta'), tf.all_variables())[0]
                            bn4_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn4/gamma'), tf.all_variables())[0]
                            print("fix_scale_en_bn3 beta/gamma:" + str(sess.run(bn3_beta)) + "/" + str(sess.run(bn3_gamma)))
                            print("fix_scale_en_bn4 beta/gamma:" + str(sess.run(bn4_beta)) + "/" + str(sess.run(bn4_gamma)))
                start += batch_size
                end += batch_size
                iterations += 1
                            #         draw pic every n epoch
            if np.mod(epoch, args.drawing_epoch) == 0:
                X_right, _ = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal)
                X_right = X_right.reshape([-1, 28, 28, 1]) / 255
                X_left = vaX[0:visualize_dim].reshape([-1, 28, 28, 1]) / 255
                X_diff, _ = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal, feature="F_I_F_D_F_V")
                X_diff = X_diff.reshape([-1, 28, 28, 1]) / 255.
                VDSN_model.is_training = False
                image_gen_left_val, F_V_matrix, F_I_matrix,\
                F_I_right_IV_right_left_gen_out_val, F_I_diff_IV_diff_left_gen_out_val, F_I_diff_V_left_gen_out_val = sess.run(
                    [image_gen_left, F_V_left, F_I_left,F_I_right_IV_right_left_gen_out, F_I_diff_IV_diff_left_gen_out, F_I_diff_V_left_gen_out],
                    feed_dict={
                        image_real_left: X_left,
                        image_real_right: X_right,
                        image_real_diff: X_diff
                    })

                # since 16 * 8  = batch size * 2
                # if args.recon_img_not_tensorboard == True:
                #     save_visualization_triplet(X_right, X_left,
                #                                image_gen_left_val,
                #                                (int(math.ceil(batch_size ** (.5))),
                #                                 int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))),
                #                                save_path=args.pic_dir_parent + time_dir + '/sample_%04d.jpg' % int(
                #                                    iterations))
                # else:
                recon_img = get_visualization_triplet(X_right, X_left,
                                                image_gen_left_val,
                                                (int(math.ceil(batch_size ** (.5))),
                                                 int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))))
                F_V_out_diff_stitch_imgs = get_visualization_triplet(X_diff, X_left,
                                                                     F_I_diff_V_left_gen_out_val,
                                                                     (int(math.ceil(batch_size ** (.5))),
                                                                      int(math.ceil(
                                                                          batch_size / math.ceil(batch_size ** (.5))))))
                print F_V_out_diff_stitch_imgs.shape
                summary = sess.run(summary_merge_recon_img, feed_dict={
                    val_recon_img: np.expand_dims(recon_img, axis=0),
                    F_V_out_diff_stitch_img: np.expand_dims(F_V_out_diff_stitch_imgs, axis=0)
                })
                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))

                if epoch >= args.recon_epoch:

                    F_IV_out_same_stitch_imgs = get_visualization_triplet(X_right, X_left,
                                                                          F_I_right_IV_right_left_gen_out_val,
                                                    (int(math.ceil(batch_size ** (.5))),
                                                     int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))))

                    F_IV_out_diff_stitch_imgs = get_visualization_triplet(X_diff, X_left,
                                                                          F_I_diff_IV_diff_left_gen_out_val,
                                                    (int(math.ceil(batch_size ** (.5))),
                                                     int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))))

                    summary = sess.run(summary_merge_stitch_img, feed_dict={
                        F_IV_out_same_stitch_img: np.expand_dims(F_IV_out_same_stitch_imgs, axis=0),
                        F_IV_out_diff_stitch_img: np.expand_dims(F_IV_out_diff_stitch_imgs, axis=0)
                    })
                    training_writer.add_summary(summary, tf.train.global_step(sess, global_step))

                if args.clustering_tsb:
                    # draw clustering every n epoch
                    for start_cluster, end_cluster in zip(
                            range(0, len(vaX), batch_size),
                            range(batch_size, len(vaY), batch_size)
                    ):
                        corrRightVal, _ = randomPickRight(start_cluster, end_cluster, vaX, vaY, indexTableVal)
                        X_left = vaX[start_cluster:end_cluster].reshape([-1, 28, 28, 1]) / 255
                        F_V_matrix, F_I_matrix = sess.run(
                            [F_V_left, F_I_left],
                            feed_dict={
                                image_real_left: X_left,
                                image_real_right: corrRightVal.reshape([-1, 28, 28, 1]) / 255
                            })
                        if start_cluster == 0:
                            image_real_left_agg = X_left
                            F_V_matrix_agg = F_V_matrix
                            F_I_matrix_agg = F_I_matrix
                        else:
                            image_real_left_agg = np.concatenate((image_real_left_agg, X_left), axis=0)
                            F_V_matrix_agg = F_V_matrix = np.concatenate((F_V_matrix_agg, F_V_matrix), axis=0)
                            F_I_matrix_agg = F_I_matrix = np.concatenate((F_I_matrix_agg, F_I_matrix), axis=0)
                    add_tensorboard(image_real_left_agg, vaY[0: len(vaY)], F_I_matrix_agg, F_V_matrix_agg,
                                      time_dir, iterations)
                VDSN_model.is_training = True
            if np.mod(epoch, args.save_epoch) == 0 and epoch > 0:
                save_path = saver.save(sess, "{}model.ckpt".format(model_dir), global_step=global_step)
                print("Model saved in file: %s" % save_path)
            np.random.shuffle(index_left)
            start, end = resetIndex()
            epoch += 1

        # Save the variables to disk.
        save_path = saver.save(sess, "{}model.ckpt".format(model_dir), global_step=global_step)
        print("Model saved in file: %s" % save_path)

    except KeyboardInterrupt:
        print("Manual interrupt occurred.")
        print('Done training for {} steps'.format(iterations))
        save_path = saver.save(sess, "{}model.ckpt".format(model_dir), global_step=global_step)
        print("Model saved in file: %s" % save_path)
        with open(training_logs_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
            json.dump(vars(args), file)
            print("dumped args info to " + training_logs_dir + 'step' + str(iterations) + '_parameter.txt')
            # file.write(json.dump(args))
        with open(model_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
            json.dump(vars(args), file)
            print("dumped args info to " + model_dir + 'step' + str(iterations) + '_parameter.txt')

F_classification_conf = {
    "save_path": save_path,
    "batch_size": batch_size,
    "dim_y": dim_y,
    "dim_W1": dim_W1,
    "dim_W2": dim_W2,
    "dim_W3": dim_W3,
    "dim_F_I": dim_F_I,
    "image_shape": [28,28,1],
    "logs_dir_root": args.logs_dir_root,
    "F_validation_logs_dir_root": args.F_V_validation_logs_dir_root,
    "F_validation_n_epochs": args.F_V_validation_n_epochs,
    "F_validation_learning_rate": args.F_V_validation_learning_rate,
    "F_validation_test_batch_size": args.F_V_validation_test_batch_size,
    "time_dir": time_dir,
    "split_encoder" : (not args.not_split_encoder),
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
    F_V_classification_conf = copy.deepcopy(F_classification_conf)
    F_V_classification_conf["feature_selection"] = "F_V"
    classification_validation.validate_F_classification(F_V_classification_conf, trX, trY, vaX, vaY, teX, teY, True)

if args.validate_classification:
    tf.reset_default_graph()
    F_I_classification_conf = copy.deepcopy(F_classification_conf)
    F_I_classification_conf["feature_selection"] = "F_I"
    classification_validation.validate_F_classification(F_I_classification_conf, trX, trY, vaX, vaY, teX, teY, True)
