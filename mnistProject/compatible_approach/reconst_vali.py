import numpy as np
import tensorflow as tf
from reconst_vali_model import *
from mnistProject.util import *
import argparse
from mnistProject.load import *
import math
import json

'''
This function would train reconstruction of F_I + zeros, F_I + cross F_V, etc,
make sure it cannot train out the Identity
'''

def validate_reconst_identity(conf, trX, trY, vaX, vaY, teX, teY):

    global image_generated_val
    check_create_dir(conf["logs_dir_root"])
    check_create_dir(conf["logs_dir_root"] + conf["F_validation_logs_dir_root"])

    reconst_vali_model = reconst_validation_model(
        batch_size=conf["batch_size"],
        image_shape=conf["image_shape"],
        dim_y=conf["dim_y"],
        dim_W1=conf["dim_W1"],
        dim_W2=conf["dim_W2"],
        dim_W3=conf["dim_W3"],
        dim_F_I=conf["dim_F_I"]
    )
    reconst_vali_model.is_training = False
    check_create_dir(args.pic_dir_parent)
    pic_dir=args.pic_dir_parent + conf["save_path"].split('/')[-2] + '/'
    check_create_dir(pic_dir)
    if args.feature_selection == 'interpolation':
        pic_dir = pic_dir + args.feature_selection + '_' + str(args.interpolation_step) + '/'
        check_create_dir(pic_dir)
    else:
        pic_dir = pic_dir + args.feature_selection + '/'
        check_create_dir(pic_dir)
    Y_tf, center_representation_tf, image_real_left_tf, \
    image_real_right_tf, image_F_I_tf, image_F_V_tf, image_generated_tf, F_I_left_tf, F_V_left_tf \
        = reconst_vali_model.build_model(conf["feature_selection"])

    global_step = tf.Variable(0, trainable=False)

    # discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
    # gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    # # include en_* and encoder_* W and b,
    # encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())
    iterations = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    teX, teY = sort_by_identity(teX, teY)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        indexTable = [[] for i in range(10)]
        for index in range(len(teY)):
            indexTable[teY[index]].append(index)
        if (len(conf["save_path"])>0):
            # Create a saver. include all vars
            saver = tf.train.Saver()
            saver.restore(sess, conf["save_path"])

        class_center_representations \
            = np.zeros((conf["dim_y"], conf["dim_W1"] - conf["dim_F_I"]))

        if conf["feature_selection"]=="F_I_C" or conf["feature_selection"]=="F_I_0":
            class_count = np.zeros(conf["dim_y"])
            if conf["feature_selection"]=="F_I_C":
                Y_tf_class, image_real_tf_class, F_I_tf_class, F_V_tf_class \
                    = reconst_vali_model.build_class_center()
                for start, end in zip(
                        range(0, len(teX), conf["batch_size"]),
                        range(conf["batch_size"], len(teX), conf["batch_size"])
                ):
                    Y_tf_class_val, F_I_tf_class_val, F_V_tf_class_val = sess.run(
                        [Y_tf_class, F_I_tf_class, F_V_tf_class],
                        feed_dict={
                            Y_tf_class: teY[start:end],
                            image_real_tf_class: teX[start:end].reshape([-1, 28, 28, 1]) / 255.
                    })
                    for i in range(F_V_tf_class_val.shape[0]):
                        class_count[int(Y_tf_class_val[i])] += 1
                        class_center_representations[int(Y_tf_class_val[i])] += F_V_tf_class_val[i]
                class_center_representations = class_center_representations / class_count[:,None]

        if conf["feature_selection"]=="interpolation":
            F_I_tf, F_V_tf, img_gen_tf = reconst_vali_model.feature_2_img()
            for i in range(len(indexTable)):
                Xs_left = teX[indexTable[i][i]].reshape([-1, 28, 28, 1]) / 255.
                F_I_left_val, F_V_left_val = sess.run(
                    [F_I_left_tf, F_V_left_tf],
                    feed_dict={image_real_left_tf: Xs_left})
                # print F_I_left_val
                for j in range(len(indexTable)):
                    Xs_right = teX[indexTable[j][-1]].reshape([-1, 28, 28, 1]) / 255.
                    F_I_right_val, F_V_right_val = sess.run(
                        [F_I_left_tf, F_V_left_tf],
                        feed_dict={image_real_left_tf: Xs_right})
                    F_I_step = np.add(F_I_right_val, -F_I_left_val) / (1. * args.interpolation_step)
                    F_V_step = np.add(F_V_right_val, - F_V_left_val) / (1. * args.interpolation_step)
                    img_matrix = []
                    for r in range(args.interpolation_step + 1):
                        img_matrix.append([])
                        for c in range(args.interpolation_step + 1):
                            img_gen_val = sess.run(
                                img_gen_tf,
                                feed_dict={F_I_tf: np.add(F_I_left_val, r * F_I_step),
                                           F_V_tf: np.add(F_V_left_val, c * F_V_step)})
                            img_matrix[r].append(img_gen_val[0])
                    save_visualization_interpolation(img_matrix,
                         save_path=pic_dir + '{}_{}.jpg'.format(indexTable[i][i], indexTable[j][-1]))
        else:
            for start, end in zip(
                    range(0, len(teX), conf["batch_size"]),
                    range(conf["batch_size"], len(teX), conf["batch_size"])
            ):
                Xs_left = teX[start:end].reshape([-1, 28, 28, 1]) / 255.

                if conf["feature_selection"]=="F_I_F_V" or conf["feature_selection"]=="F_I_F_D_F_V":
                    Xs_right, _ = randomPickRight(start, end, teX, teY, indexTable,
                        feature=conf["feature_selection"])
                    Xs_right = Xs_right.reshape([-1, 28, 28, 1]) / 255.

                    image_F_I_val, image_F_V_val, image_generated_val = sess.run(
                        [image_F_I_tf, image_F_V_tf, image_generated_tf],
                        feed_dict={
                            image_real_left_tf: Xs_left,
                            image_real_right_tf: Xs_right
                            })
                    save_visualization_triplet(image_F_I_val, image_F_V_val, image_generated_val,
                       (int(math.ceil(conf['batch_size'] ** (.5))),
                        int(math.ceil(conf['batch_size'] / math.ceil(conf['batch_size'] ** (.5))))),
                            save_path=pic_dir + '%04d.jpg' % int(
                                                   iterations))
                else:
                    center_representation = class_center_representations[teY[start:end]]
                    image_F_I_val, image_F_V_val, image_generated_val = sess.run(
                        [image_F_I_tf, image_F_V_tf, image_generated_tf],
                        feed_dict={
                            image_real_left_tf: Xs_left,
                            center_representation_tf: center_representation
                        })
                    save_visualization_triplet(image_F_I_val, image_F_V_val, image_generated_val,
                       (int(math.ceil(conf['batch_size'] ** (.5))),
                        int(math.ceil(conf['batch_size'] / math.ceil(conf['batch_size'] ** (.5))))),
                       save_path=pic_dir  + '%04d.jpg' % int(
                           iterations))
                iterations += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--save_path", nargs='?', type=str, default='',
                        help="root dir to save training summary")

    parser.add_argument("--feature_selection", nargs='?', type=str, default='F_I_0',
                        help="to validate F_I_0 for F_I and 00," +
                             " or F_I_F_V for F_I and F_V,  " +
                             "F_I_C for F_I and F center " +
                             "F_I_F_D_F_V for F_I with F_V from different digit" +
                             "interpolation for F_I,F_V interpolation"
                        )

    parser.add_argument("--logs_dir_root", nargs='?', type=str, default='tensorflow_log/',
                        help="root dir to save training summary")

    parser.add_argument("--dis_regularizer_weight", nargs='?', type=float, default=0.01,
                        help="discriminator regularization weight")

    parser.add_argument("--F_validation_logs_dir_root", nargs='?', type=str, default='F_validation/',
                        help="root dir inside logs_dir_root to save F_validation summary")

    parser.add_argument("--validate_disentanglement", action="store_true",
                        help="run F disentanglement classification task")

    parser.add_argument("--F_validation_n_epochs", nargs='?', type=int, default=100,
                        help="number of epochs for F_validation")

    parser.add_argument("--F_validation_learning_rate", nargs='?', type=float, default=0.0002,
                        help="learning rate for F_validation")

    parser.add_argument("--F_validation_test_batch_size", nargs='?', type=int, default=1000,
                        help="F validation's test_batch_size")

    parser.add_argument("--gpu_ind", nargs='?', type=str, default='0',
                        help="which gpu to use")

    parser.add_argument("--time_dir", nargs='?', type=str, default='',
                        help="time dir for tensorboard")

    parser.add_argument("--pic_dir_parent", nargs='?', type=str, default='./recon_vis/',
                        help="picture folder for reconstruction validation")

    parser.add_argument("--interpolation_step", nargs='?', type=int, default=5,
                        help="number of step interval for interpolation")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind

    # amount of class label
    trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

    conf = {
        "save_path": args.save_path,
        "batch_size": args.batch_size,
        "F_validation_test_batch_size": args.F_validation_test_batch_size,
        "image_shape": [28, 28, 1],
        "dim_y": args.dim_y,
        "dim_W1": args.dim_W1,
        "dim_W2": args.dim_W2,
        "dim_W3": args.dim_W3,
        "dim_F_I": args.dim_F_I,
        "dis_regularizer_weight": args.dis_regularizer_weight,
        "logs_dir_root": args.logs_dir_root,
        "F_validation_logs_dir_root": args.F_validation_logs_dir_root,
        "F_validation_n_epochs": args.F_validation_n_epochs,
        "F_validation_learning_rate": args.F_validation_learning_rate,
        "time_dir": args.time_dir,
        "feature_selection" : args.feature_selection,
        "pic_dir_parent": args.pic_dir_parent
    }
    validate_reconst_identity(conf, trX, trY, vaX, vaY, teX, teY)
