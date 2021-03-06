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


parser.add_argument("--gen_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--dis_start_learning_rate", nargs='?', type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--gan_start_learning_rate", nargs='?', type=float, default=0.002,
                    help="learning rate")

parser.add_argument("--gen_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--dis_decay_step", nargs='?', type=int, default=10000,
                    help="generator decay step")

parser.add_argument("--gen_decay_rate", nargs='?', type=float, default=0.80,
                    help="generator decay rate")

parser.add_argument("--dis_decay_rate", nargs='?', type=float, default=1.00,
                    help="discriminator decay rate")

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

parser.add_argument("--n_epochs", nargs='?', type=int, default=100,
                    help="number of epochs")

parser.add_argument("--recon_series", nargs='?', type=int, default=1,
                    help="how many time the generator with reconstruction task can train consecutively")

parser.add_argument("--dis_series", nargs='?', type=int, default=10,
                    help="how many time the adversarial dis can train consecutively")

parser.add_argument("--gan_series", nargs='?', type=int, default=1,
                    help="how many time the generator with gan task can train consecutively")

parser.add_argument("--drawing_step", nargs='?', type=int, default=100000,
                    help="how many steps to draw a comparision pic")

parser.add_argument("--save_step", nargs='?', type=int, default=200000,
                    help="how many steps to save")

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

parser.add_argument("--recon_img_not_tensorboard", action="store_true",
                    help="save validation reconstruction image locally not tensorboard")

parser.add_argument("--gpu_ind", nargs='?', type=str, default='0',
                    help="which gpu to use")

parser.add_argument("--disentangle_obj_func", nargs='?', type=str, default='negative_log',
                    help="generator's disentanglement loss use which loss, negative_log, one_minus, hybrid or complex")

parser.add_argument("--debug", action="store_true",
                    help="debug_mode")

parser.add_argument("--train_bn", action="store_true",
                    help="train bn's gamma and beta")

parser.add_argument("--soft_bn", action="store_true",
                    help="use moving in training")

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
print time_dir
gen_disentangle_weight = args.gen_disentangle_weight
gen_regularizer_weight = args.gen_regularizer_weight
dis_regularizer_weight = args.dis_regularizer_weight
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
        disentangle_obj_func=args.disentangle_obj_func,
        train_bn= args.train_bn,
        soft_bn= args.soft_bn
)

Y_left_tf, Y_right_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf, gen_disentangle_cost_tf, gen_cla_cost_tf,\
    gen_total_cost_tf, dis_cost_tf, dis_total_cost_tf, \
    image_gen_left, image_gen_right, dis_prediction_tf_left, dis_prediction_tf_right, gen_cla_accuracy_tf, \
    F_I_left_tf, F_V_left_tf, gan_gen_cost_tf, gan_dis_cost_tf, gan_total_cost_tf, val_recon_img_tf, \
    summary_merge_scalar, summary_merge_img \
    = VDSN_model.build_model(
    gen_disentangle_weight, gen_regularizer_weight, dis_regularizer_weight, gen_cla_weight)

global_step = tf.Variable(0, trainable=False)
# saver to save trained model to disk
saver = tf.train.Saver(max_to_keep=10)
# global_step to record steps in total
gen_learning_rate = tf.train.exponential_decay(args.gen_start_learning_rate, global_step,
                                               args.gen_decay_step, args.gen_decay_rate, staircase=True)
dis_learning_rate = tf.train.exponential_decay(args.dis_start_learning_rate, global_step,
                                               args.dis_decay_step, args.dis_decay_rate, staircase=True)
gan_learning_rate = tf.train.exponential_decay(args.gan_start_learning_rate, global_step,
                                               args.gan_decay_step, args.gan_decay_rate, staircase=True)

if args.train_bn:
    dis_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    cla_vars = filter(lambda x: x.name.startswith('cla'), tf.trainable_variables())
    gan_dis_vars = filter(lambda x: x.name.startswith('gan'), tf.trainable_variables())
    # include en_* and encoder_* W and b,
    encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())
else:
    dis_vars = filter(lambda x: x.name.startswith('dis') and 'bn' not in x.name, tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen') and 'bn' not in x.name, tf.trainable_variables())
    cla_vars = filter(lambda x: x.name.startswith('cla') and 'bn' not in x.name, tf.trainable_variables())
    gan_dis_vars = filter(lambda x: x.name.startswith('gan') and 'bn' not in x.name, tf.trainable_variables())
    # include en_* and encoder_* W and b,
    encoder_vars = filter(lambda x: x.name.startswith('en') and 'bn' not in x.name, tf.trainable_variables())

with tf.control_dependencies(tf.get_collection(GEN_BATCH_NORM_OPS)):
    train_op_gen = tf.train.AdamOptimizer(
        gen_learning_rate, beta1=0.5).minimize(
        gen_total_cost_tf, var_list=gen_vars+encoder_vars+cla_vars, global_step=global_step)

with tf.control_dependencies(tf.get_collection(ADV_BATCH_NORM_OPS)):
    train_op_discrim = tf.train.AdamOptimizer(
        dis_learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=dis_vars)

with tf.control_dependencies(tf.get_collection(GEN_BATCH_NORM_OPS)):
    train_op_gan_gen = tf.train.AdamOptimizer(
        gan_learning_rate, beta1=0.5).minimize(
        gan_total_cost_tf, var_list=gen_vars+encoder_vars+cla_vars)

with tf.control_dependencies(tf.get_collection(DIS_BATCH_NORM_OPS)):
    # not update global step since we considered gan as a whole step
    train_op_gan_discrim = tf.train.AdamOptimizer(
        gan_learning_rate, beta1=0.5).minimize(
        gan_dis_cost_tf, var_list=gan_dis_vars)

iterations = 0
save_path=""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def resetIndex():
    start = 0
    end = args.batch_size
    return start, end

with tf.Session(config=config) as sess:
    try:
        sess.run(tf.global_variables_initializer())
        # writer for tensorboard summary
        training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
        # test_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)


        if (len(args.pretrain_model)>0):
            # Create a saver. include gen_vars and encoder_vars
            saver.restore(sess, args.pretrain_model)
        elif (len(args.pretrain_model_wo_lr)>0):
            # Create a saver. include gen_vars and encoder_vars
            pretrain_saver = tf.train.Saver(gen_vars + encoder_vars + dis_vars + cla_vars)
            pretrain_saver.restore(sess, args.pretrain_model_wo_lr)

        index_disentangle = np.arange(len(trY))
        index_reconst = np.arange(len(trY))
        index_gan = np.arange(len(trY))

        indexTable = [[] for i in range(dim_y)]
        for index in range(len(trY)):
            indexTable[trY[index]].append(index)
        indexTableVal = [[] for i in range(10)]
        for index in range(len(vaY)):
            indexTableVal[vaY[index]].append(index)
        start_disentangle, end_disentangle = resetIndex()
        start_reconst, end_reconst = resetIndex()
        start_gan, end_gan = resetIndex()
        np.random.shuffle(index_disentangle)
        np.random.shuffle(index_reconst)
        np.random.shuffle(index_gan)

        epoch = 0
        while epoch < n_epochs:
            while end_reconst <= len(trY):
                print epoch
                if end_disentangle > len(trY):
                    np.random.shuffle(index_disentangle)
                    start_disentangle, end_disentangle = resetIndex()
                if end_gan > len(trY):
                    np.random.shuffle(index_gan)
                    start_gan, end_gan = resetIndex()

                Xs_left_disentangle = trX[index_disentangle[start_disentangle:end_disentangle]].reshape([-1, 28, 28, 1]) / 255.
                Ys_left_disentangle = OneHot(trY[index_disentangle[start_disentangle:end_disentangle]], 10)
                Xs_left_reconst = trX[index_reconst[start_reconst:end_reconst]].reshape([-1, 28, 28, 1]) / 255.
                Ys_left_reconst = OneHot(trY[index_reconst[start_reconst:end_reconst]], 10)
                Xs_left_gan = trX[index_gan[start_gan:end_gan]].reshape([-1, 28, 28, 1]) / 255.
                Ys_left_gan = OneHot(trY[index_gan[start_gan:end_gan]], 10)
                Xs_right = []
                Ys_right = []

                modulus = np.mod(iterations, args.gan_series + args.dis_series + args.recon_series)

                if modulus < args.recon_series:
                    Xs_right, Ys_right = randomPickRight(start_reconst, end_reconst, trX,
                                                         trY[index_reconst[start_reconst:end_reconst]], indexTable)
                    Xs_right = Xs_right.reshape([-1, 28, 28, 1]) / 255.
                elif modulus < args.recon_series + args.dis_series:
                    Xs_right, Ys_right = randomPickRight(start_disentangle, end_disentangle, trX,
                                                         trY[index_disentangle[start_disentangle:end_disentangle]],
                                                         indexTable)
                    Xs_right = Xs_right.reshape([-1, 28, 28, 1]) / 255.
                else:
                    Xs_right, Ys_right = randomPickRight(start_gan, end_gan, trX,
                                                         trY[index_gan[start_gan:end_gan]], indexTable,
                                                         feature="F_I_F_D_F_V")
                    Ys_right = OneHot(Ys_right, 10)
                    Xs_right = Xs_right.reshape([-1, 28, 28, 1]) / 255.
                if modulus < args.recon_series:
                    _, summary, gen_recon_cost_val, gen_disentangle_val, gen_cla_cost_val, gen_total_cost_val, \
                            dis_prediction_val_left, dis_prediction_val_right, gen_cla_accuracy_val \
                        = sess.run(
                            [train_op_gen, summary_merge_scalar, g_recon_cost_tf,
                             gen_disentangle_cost_tf, gen_cla_cost_tf, gen_total_cost_tf,
                             dis_prediction_tf_left, dis_prediction_tf_right,gen_cla_accuracy_tf],
                            feed_dict={
                                Y_left_tf:Ys_left_reconst,
                                Y_right_tf:Ys_left_reconst,
                                image_tf_real_left: Xs_left_reconst,
                                image_tf_real_right: Xs_right
                            })
                    start_reconst += batch_size
                    end_reconst += batch_size
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
                    if args.debug:
                        bn3_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn3/beta'), tf.all_variables())[0]
                        bn3_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn3/gamma'), tf.all_variables())[0]
                        bn4_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn4/beta'), tf.all_variables())[0]
                        bn4_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn4/gamma'), tf.all_variables())[0]
                        print("fix_scale_en_bn3 beta/gamma:" + str(sess.run(bn3_beta)) + "/"+str(sess.run(bn3_gamma)))
                        print("fix_scale_en_bn4 beta/gamma:" + str(sess.run(bn4_beta)) + "/"+str(sess.run(bn4_gamma)))
                elif modulus < args.recon_series + args.dis_series:
                    _, summary, dis_cost_val, dis_total_cost_val, \
                            dis_prediction_val_left, dis_prediction_val_right \
                        = sess.run(
                            [train_op_discrim, summary_merge_scalar, dis_cost_tf, dis_total_cost_tf, \
                             dis_prediction_tf_left, dis_prediction_tf_right],
                            feed_dict={
                                Y_left_tf:Ys_left_disentangle,
                                Y_right_tf:Ys_left_disentangle,
                                image_tf_real_left: Xs_left_disentangle,
                                image_tf_real_right: Xs_right
                                })
                    training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    start_disentangle += batch_size
                    end_disentangle += batch_size
                    print("=========== updating D ==========")
                    print("iteration:", iterations)
                    print("discriminator loss:", dis_cost_val)
                    print("discriminator total weigthted loss:", dis_total_cost_val)
                    print("discrim left correct prediction's max,mean,min :", dis_prediction_val_left)
                    print("discrim right correct prediction's max,mean,min :", dis_prediction_val_right)
                    # moving_mean = filter(lambda x: x.name.startswith('dis_bn1/moving_mean'), tf.all_variables())[0]
                    # print(moving_mean.name + ":" + str(sess.run(moving_mean)))

                else:
                    # start to train gan, D first
                    _, summary, gan_dis_cost \
                        = sess.run(
                        [train_op_gan_discrim, summary_merge_scalar, gan_dis_cost_tf],
                        feed_dict={
                            Y_left_tf: Ys_left_gan,
                            Y_right_tf: Ys_right,
                            image_tf_real_left: Xs_left_gan,
                            image_tf_real_right: Xs_right
                        }
                    )
                    print("=========== updating gan D ==========")
                    print("iteration:", iterations)
                    print("gan_dis_cost:", gan_dis_cost)
                    # moving_mean = filter(lambda x: x.name.startswith('gan_dis_bn1/moving_mean'), tf.all_variables())[0]
                    # print(moving_mean.name + ":" + str(sess.run(moving_mean)))
                    # train G
                    _, summary, gan_gen_cost_val, gen_disentangle_val, gen_cla_cost_val, gan_total_cost_val, \
                    dis_prediction_val_left, dis_prediction_val_right, gen_cla_accuracy_val \
                        = sess.run(
                        [train_op_gan_gen, summary_merge_scalar, gan_gen_cost_tf,
                         gen_disentangle_cost_tf, gen_cla_cost_tf, gan_total_cost_tf,
                         dis_prediction_tf_left, dis_prediction_tf_right, gen_cla_accuracy_tf],
                        feed_dict={
                            Y_left_tf: Ys_left_gan,
                            Y_right_tf: Ys_right,
                            image_tf_real_left: Xs_left_gan,
                            image_tf_real_right: Xs_right
                    })
                    start_gan += batch_size
                    end_gan += batch_size
                    print("=========== updating gan G ==========")
                    print("iteration:", iterations)
                    print("gan gen loss:", gan_gen_cost_val)
                    print("gen disentanglement loss :", gen_disentangle_val)
                    print("gen id classifier loss :", gen_cla_cost_val)
                    print("total weigthted gan loss :", gan_total_cost_val)
                    print("discrim left correct prediction's max,mean,min:", dis_prediction_val_left)
                    print("discrim right correct prediction's max,mean,min:", dis_prediction_val_right)
                    print("gen id classifier accuracy:", gen_cla_accuracy_val)
                    if args.debug:
                        bn3_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn3/beta'), tf.all_variables())[0]
                        bn3_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn3/gamma'), tf.all_variables())[0]
                        bn4_beta = filter(lambda x: x.name.startswith('fix_scale_en_bn4/beta'), tf.all_variables())[0]
                        bn4_gamma = filter(lambda x: x.name.startswith('fix_scale_en_bn4/gamma'), tf.all_variables())[0]
                        print("fix_scale_en_bn3 beta/gamma:" + str(sess.run(bn3_beta)) + "/" + str(sess.run(bn3_gamma)))
                        print("fix_scale_en_bn4 beta/gamma:" + str(sess.run(bn4_beta)) + "/" + str(sess.run(bn4_gamma)))
                if np.mod(iterations, drawing_step) == 0:
                    corrRightVal, _ = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal)
                    corrRightVal = corrRightVal.reshape([-1, 28, 28, 1]) / 255
                    image_real_left = vaX[0:visualize_dim].reshape([-1, 28, 28, 1]) / 255
                    VDSN_model.is_training = False
                    generated_samples_left, F_V_matrix, F_I_matrix = sess.run(
                        [image_gen_left, F_V_left_tf, F_I_left_tf],
                        feed_dict={
                            image_tf_real_left: image_real_left,
                            image_tf_real_right: corrRightVal
                        })
                    # since 16 * 8  = batch size * 2
                    if args.recon_img_not_tensorboard == True:
                        save_visualization_triplet(corrRightVal,image_real_left,
                                                   generated_samples_left,
                                                   (int(math.ceil(batch_size ** (.5))),
                                                    int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))),
                                                   save_path=args.pic_dir_parent + time_dir + '/sample_%04d.jpg' % int(
                                                       iterations))
                    else:
                        img = get_visualization_triplet(corrRightVal,image_real_left,
                                                   generated_samples_left,
                                                   (int(math.ceil(batch_size ** (.5))),
                                                    int(math.ceil(batch_size / math.ceil(batch_size ** (.5))))))
                        summary = sess.run(summary_merge_img,feed_dict={
                            val_recon_img_tf: np.expand_dims(img, axis=0)})
                        training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    VDSN_model.is_training = True
                iterations += 1
                if np.mod(iterations, args.save_step) == 0 and iterations >= args.save_step:
                    save_path = saver.save(sess, "{}model.ckpt".format(model_dir, global_step=global_step))
                    print("Model saved in file: %s" % save_path)

            np.random.shuffle(index_reconst)
            start_reconst, end_reconst = resetIndex()
            epoch += 1

        # Save the variables to disk.
        save_path = saver.save(sess, "{}model.ckpt".format(model_dir, global_step=global_step))
        print("Model saved in file: %s" % save_path)

    except KeyboardInterrupt:
        print("Manual interrupt occurred.")
        print('Done training for {} steps'.format(iterations))
        save_path = saver.save(sess, "{}model.ckpt".format(model_dir, global_step=global_step))
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
    "dis_regularizer_weight": args.dis_regularizer_weight,
    "logs_dir_root": args.logs_dir_root,
    "F_validation_logs_dir_root": args.F_V_validation_logs_dir_root,
    "F_validation_n_epochs": args.F_V_validation_n_epochs,
    "F_validation_learning_rate": args.F_V_validation_learning_rate,
    "F_validation_test_batch_size": args.F_V_validation_test_batch_size,
    "time_dir": time_dir,
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
