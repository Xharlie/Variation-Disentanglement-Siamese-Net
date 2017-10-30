import numpy as np
import tensorflow as tf
from reconst_vali_model import *
from util import *
import argparse
from load import *
import math
import json

'''
This function would train a classifier on top of the representation F_V,
make sure it cannot train out the Identity
'''

def validate_reconst_identity(conf):

    check_create_dir(conf["logs_dir_root"])
    check_create_dir(conf["logs_dir_root"] + conf["F_V_validation_logs_dir_root"])
    training_logs_dir = check_create_dir(conf["logs_dir_root"]
                      + conf["F_V_validation_logs_dir_root"]+conf["time_dir"]+'/')

    # test_logs_dir = check_create_dir(conf["logs_dir_root"]
    #                  + conf["F_V_validation_logs_dir_root"]+'test/')

    reconst_vali_model = reconst_validation_model(
        batch_size=conf["batch_size"],
        image_shape=conf["image_shape"],
        dim_y=conf["dim_y"],
        dim_W1=conf["dim_W1"],
        dim_W2=conf["dim_W2"],
        dim_W3=conf["dim_W3"],
        dim_F_I=conf["dim_F_I"]
    )

    Y_tf, image_real_tf, image_gen_tf \
        = reconst_vali_model.build_model()

    global_step = tf.Variable(0, trainable=False)

    discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    # include en_* and encoder_* W and b,
    encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())
    iterations = 0
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        if (len(conf["save_path"])>0):
            # Create a saver. include gen_vars and encoder_vars
            saver = tf.train.Saver(gen_vars + encoder_vars)
            saver.restore(sess, conf["save_path"])

        teX = conf["teX"]

        for start, end in zip(
                range(0, len(teX), conf["batch_size"]),
                range(conf["batch_size"], len(teX), conf["batch_size"])
        ):
            image_real_left = teX[start:end].reshape([-1, 28, 28, 1]) / 255
            generated_samples = sess.run(
                    image_gen_tf,
                    feed_dict={
                        image_real_tf: image_real_left
                        })
            save_visualization(image_real_left, generated_samples,
                               (int(math.ceil(conf['batch_size'] ** (.5))),
                                int(math.ceil(conf['batch_size'] / math.ceil(conf['batch_size'] ** (.5))))),
                               save_path=args.pic_dir_parent + 'sample_reconst_mean_%04d.jpg' % int(iterations))
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

    parser.add_argument("--logs_dir_root", nargs='?', type=str, default='tensorflow_log/',
                        help="root dir to save training summary")

    parser.add_argument("--dis_regularizer_weight", nargs='?', type=float, default=0.01,
                        help="discriminator regularization weight")

    parser.add_argument("--pic_dir_parent", nargs='?', type=str, default='recon_vis/',
                        help="root dir to save pic")

    parser.add_argument("--drawing_step", nargs='?', type=int, default=200,
                        help="how many steps to draw a comparision pic")

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

    parser.add_argument("--gpu_ind", nargs='?', type=str, default='0',
                        help="which gpu to use")

    parser.add_argument("--time_dir", nargs='?', type=str, default='',
                        help="time dir for tensorboard")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind

    # amount of class label
    trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

    F_I_classification_conf = {
        "save_path": args.save_path,
        "trX": trX,
        "trY": trY,
        "vaX": vaX,
        "vaY": vaY,
        "teX": teX,
        "teY": teY,
        "batch_size": args.batch_size,
        "F_V_validation_test_batch_size": args.F_V_validation_test_batch_size,
        "image_shape": [28, 28, 1],
        "dim_y": args.dim_y,
        "dim_W1": args.dim_W1,
        "dim_W2": args.dim_W2,
        "dim_W3": args.dim_W3,
        "dim_F_I": args.dim_F_I,
        "dis_regularizer_weight": args.dis_regularizer_weight,
        "logs_dir_root": args.logs_dir_root,
        "F_V_validation_logs_dir_root": args.F_V_validation_logs_dir_root,
        "F_V_validation_n_epochs": args.F_V_validation_n_epochs,
        "F_V_validation_learning_rate": args.F_V_validation_learning_rate,
        "time_dir": args.time_dir
    }
    validate_reconst_identity(F_I_classification_conf)
