from compatible_approach.model import *
from util import *
from cluster import *
from load import mnist_with_valid_set
from time import localtime, strftime
import argparse

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

parser.add_argument("--batch_size", nargs='?', type=int, default=64,
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

parser.add_argument("--recon_series", nargs='?', type=int, default=1,
                    help="how many time the generator with reconstruction task can train consecutively")

parser.add_argument("--gan_series", nargs='?', type=int, default=1,
                    help="how many time the generator with gan task can train consecutively")

parser.add_argument("--drawing_step", nargs='?', type=int, default=100000,
                    help="how many steps to draw a comparision pic")

parser.add_argument("--save_step", nargs='?', type=int, default=200000,
                    help="how many steps to save")

parser.add_argument("--gen_regularizer_weight", nargs='?', type=float, default=0.01,
                    help="generator regularization weight")

parser.add_argument("--gen_metric_loss_weight", nargs='?', type=float, default=10.0,
                    help="metric loss")

parser.add_argument("--gen_cla_weight", nargs='?', type=float, default=5.0,
                    help="generator classifier weight")

parser.add_argument("--logs_dir_root", nargs='?', type=str, default='tensorflow_log/',
                    help="root dir to save training summary")

parser.add_argument("--main_logs_dir_root", nargs='?', type=str, default='main/',
                    help="root dir inside logs_dir_root to save main summary")

parser.add_argument("--pretrain_model_time_dir", nargs='?', type=str, default='',
                    help="pretrain model time dir ")

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

parser.add_argument("--gan_gen_weight", nargs='?', type=float, default=1.0,
                    help="gan_gen_weight weight")

parser.add_argument("--F_IV_recon_weight", nargs='?', type=float, default=1.0,
                    help="F_IV_recon_weight weight")

parser.add_argument("--F_I_gen_recon_weight", nargs='?', type=float, default=1.0,
                    help="F_I_gen_recon_weight weight")

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
gen_regularizer_weight = args.gen_regularizer_weight
dis_regularizer_weight = args.dis_regularizer_weight
gen_cla_weight = args.gen_cla_weight
# if we don't have these directory, create them
check_create_dir(args.logs_dir_root)
check_create_dir(args.logs_dir_root + args.main_logs_dir_root)
check_create_dir(args.model_dir_parent)
check_create_dir(args.pic_dir_parent)

training_logs_dir = check_create_dir(args.logs_dir_root + args.main_logs_dir_root + time_dir + '/')
model_dir = check_create_dir('compatible_approach/' + args.model_dir_parent + '' +args.pretrain_model_time_dir + '/')
model_file = model_dir+'model.ckpt'
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
        metric_obj_func=args.metric_obj_func,
        train_bn= args.train_bn,
        soft_bn= args.soft_bn
)

Y_left, Y_right, Y_diff, image_real_left, image_real_right, image_real_diff, gen_recon_cost, gen_metric_loss, \
   gen_cla_cost, gen_total_cost, image_gen_left, image_gen_right, gen_cla_accuracy, F_I_left, F_V_left, \
   gan_gen_cost, F_IV_recon_cost, gan_dis_total_cost, gan_gen_total_cost, val_recon_img, F_I_cluster_img, F_V_cluster_img, \
   F_I_center_left, F_I_center_right, F_I_center_diff, F_I_right, F_V_right, F_I_right_IV_right_left_gen_out, \
   F_I_diff_IV_diff_left_gen_out,F_I_diff_V_left_gen_out, F_IV_out_diff_stitch_img, F_IV_out_same_stitch_img, \
   F_V_out_diff_stitch_img, summary_merge_scalar, summary_gen_merge_scalar,summary_gan_gen_merge_scalar, \
   summary_gan_dis_merge_scalar, summary_merge_recon_img, summary_merge_cluster_img, summary_merge_stitch_img \
    = VDSN_model.build_model(
    args.gen_metric_loss_weight, gen_regularizer_weight, gen_cla_weight, args.gan_gen_weight, args.F_IV_recon_weight, args.F_I_gen_recon_weight)

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
with tf.Session(config=tf.ConfigProto()) as sess:

    sess.run(tf.global_variables_initializer())
    pretrain_saver = tf.train.Saver(encoder_vars)
    pretrain_saver.restore(sess, model_file)
    image_real_left_agg = None
    F_V_matrix_agg = None
    F_I_matrix_agg = None
    indexTableVal = [[] for i in range(10)]
    for index in range(len(teY)):
        indexTableVal[teY[index]].append(index)
    for start, end in zip(
            range(0, len(teX), batch_size),
            range(batch_size, len(teY), batch_size)
            ):
        corrRightVal,_ = randomPickRight(start, end, teX, teY, indexTableVal)
        X_left = teX[start:end].reshape([-1, 28, 28, 1]) / 255
        F_V_matrix, F_I_matrix = sess.run(
            [F_V_left, F_I_left],
            feed_dict={
                image_real_left: X_left,
                image_real_right: corrRightVal.reshape([-1, 28, 28, 1]) / 255
            })
        if start == 0:
            image_real_left_agg = X_left
            F_V_matrix_agg = F_V_matrix
            F_I_matrix_agg = F_I_matrix
        else:
            image_real_left_agg = np.concatenate((image_real_left_agg, X_left), axis=0)
            F_V_matrix_agg = F_V_matrix = np.concatenate((F_V_matrix_agg, F_V_matrix), axis=0)
            F_I_matrix_agg = F_I_matrix = np.concatenate((F_I_matrix_agg, F_I_matrix), axis=0)
        iterations += 1
    cluster(image_real_left_agg, teY[0: len(teY)], F_I_matrix_agg, F_V_matrix_agg, args.pretrain_model_time_dir, iterations, is_tensorboard=False)

    # cluster(image_real_left_agg, teY[0: len(teY)], F_I_matrix_agg, F_V_matrix_agg,
    #         args.pretrain_model_time_dir, iterations)
