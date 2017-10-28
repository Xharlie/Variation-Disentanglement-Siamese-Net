import numpy as np
from F_V_validation_model import *
from util import *
import argparse
from load import *
import json

'''
This function would train a classifier on top of the representation F_V,
make sure it cannot train out the Identity
'''

def validate_F_V_classification_fail(conf):

    check_create_dir(conf["logs_dir_root"])
    check_create_dir(conf["logs_dir_root"] + conf["F_V_validation_logs_dir_root"])
    training_logs_dir = check_create_dir(conf["logs_dir_root"]
                      + conf["F_V_validation_logs_dir_root"]+conf["time_dir"]+'/')

    # test_logs_dir = check_create_dir(conf["logs_dir_root"]
    #                  + conf["F_V_validation_logs_dir_root"]+'test/')

    F_V_validation_model = F_V_validation(
        batch_size=conf["batch_size"],
        image_shape=conf["image_shape"],
        dim_y=conf["dim_y"],
        dim_W1=conf["dim_W1"],
        dim_W2=conf["dim_W2"],
        dim_W3=conf["dim_W3"],
        dim_F_I=conf["dim_F_I"]
    )

    Y_tf, image_real_tf, dis_cost_tf, dis_total_cost_tf, Y_prediction_prob_tf,accuracy_tf \
        = F_V_validation_model.build_model(conf["dis_regularizer_weight"])

    global_step = tf.Variable(0, trainable=False)

    discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    # include en_* and encoder_* W and b,
    encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())

    train_op = tf.train.AdamOptimizer(conf["F_V_validation_learning_rate"], beta1=0.5) \
        .minimize(dis_total_cost_tf, var_list=discrim_vars, global_step=global_step)
    iterations = 0

    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
        # test_writer = tf.summary.FileWriter(test_logs_dir, sess.graph)
        train_merged_summary = tf.summary.merge_all('train')
        validation_merged_summary = tf.summary.merge_all('validation')
        test_merged_summary = tf.summary.merge_all('test')

        if (len(conf["save_path"])>0):
            # Create a saver. include gen_vars and encoder_vars
            saver = tf.train.Saver(gen_vars + encoder_vars)
            saver.restore(sess, conf["save_path"])

        trX = conf["trX"]
        trY = conf["trY"]
        vaX = conf["vaX"]
        vaY = conf["vaY"]
        teX = conf["teX"]
        teY = conf["teY"]

        for epoch in range(conf["F_V_validation_n_epochs"]):
            index = np.arange(len(conf["trY"]))
            np.random.shuffle(index)
            trX = trX[index]
            trY = trY[index]

            for start, end in zip(
                    range(0, len(trY), conf["batch_size"]),
                    range(conf["batch_size"], len(trY), conf["batch_size"])
            ):
                # pixel value normalized -> from 0 to 1
                Xs = trX[start:end].reshape([-1, 28, 28, 1]) / 255.
                Ys = OneHot(trY[start:end], 10)

                _, summary, dis_cost_val, dis_total_cost_val, Y_prediction_prob_val,accuracy_val \
                    = sess.run(
                        [train_op, train_merged_summary, dis_cost_tf,
                         dis_total_cost_tf, Y_prediction_prob_tf,accuracy_tf],
                        feed_dict={
                            Y_tf: Ys,
                            image_real_tf: Xs,
                        })
                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))

                print("=========== iteration: ==========",iterations)
                print("train discriminator loss:", dis_cost_val)
                print("train discriminator total weigthted loss:", dis_total_cost_val)
                print("train discriminator accuracy:", accuracy_val)
                iterations = iterations + 1

            ''' validation phase each epoch '''
            dis_cost_val_list=[]
            dis_total_cost_val_list=[]
            accuracy_val_list=[]
            for start, end in zip(
                    range(0, len(vaY), conf["F_V_validation_test_batch_size"]),
                    range(conf["F_V_validation_test_batch_size"], len(vaY),  \
                    conf["F_V_validation_test_batch_size"])
            ):
                Xs = vaX[start:end].reshape([-1, 28, 28, 1]) / 255.
                Ys = OneHot(vaY[start:end], 10)

                summary, dis_cost_val, dis_total_cost_val, Y_prediction_prob_val,accuracy_val \
                    = sess.run(
                    [validation_merged_summary, dis_cost_tf,
                     dis_total_cost_tf, Y_prediction_prob_tf,accuracy_tf],
                    feed_dict={
                        Y_tf: Ys,
                        image_real_tf: Xs,
                    })
                dis_cost_val_list.append(dis_cost_val)
                dis_total_cost_val_list.append(dis_total_cost_val)
                accuracy_val_list.append(accuracy_val)
            dis_cost_val = sum(dis_cost_val_list) / len(dis_cost_val_list)
            dis_total_cost_val = sum(dis_total_cost_val_list)/len(dis_total_cost_val_list)
            accuracy_val = sum(accuracy_val_list) / len(accuracy_val_list)
            print("=========== iteration: ==========", iterations)
            print("validation discriminator loss:", dis_cost_val)
            print("validation discriminator total weigthted loss:",dis_total_cost_val)
            print("validation discriminator accuracy:", accuracy_val)
            valdiation_dis_cost_val_summary = tf.Summary(
                value=[tf.Summary.Value(tag="validation_dis_cost", simple_value=dis_cost_val)])
            training_writer.add_summary(valdiation_dis_cost_val_summary,
                                        tf.train.global_step(sess, global_step))
            valdiation_dis_total_cost_val_summary = tf.Summary(
                value=[tf.Summary.Value(tag="validation_dis_total_cost", simple_value=dis_total_cost_val)])
            training_writer.add_summary(valdiation_dis_total_cost_val_summary,
                                        tf.train.global_step(sess, global_step))
            valdiation_accuracy_val_summary = tf.Summary(
                value=[tf.Summary.Value(tag="validation_accuracy", simple_value=accuracy_val)])
            training_writer.add_summary(valdiation_accuracy_val_summary,
                                        tf.train.global_step(sess, global_step))

        ''' test phase at the end '''

        dis_cost_val_list = []
        dis_total_cost_val_list = []
        accuracy_val_list = []
        for start, end in zip(
                range(0, len(teY), conf["F_V_validation_test_batch_size"]),
                range(conf["F_V_validation_test_batch_size"], len(teY), conf["F_V_validation_test_batch_size"])
        ):
            Xs = teX[start:end].reshape([-1, 28, 28, 1]) / 255.
            Ys = OneHot(teY[start:end], 10)

            summary, dis_cost_val, dis_total_cost_val, Y_prediction_prob_val, accuracy_val \
                = sess.run(
                [test_merged_summary, dis_cost_tf,
                 dis_total_cost_tf, Y_prediction_prob_tf, accuracy_tf],
                feed_dict={
                    Y_tf: Ys,
                    image_real_tf: Xs,
                })
            dis_cost_val_list.append(dis_cost_val)
            dis_total_cost_val_list.append(dis_total_cost_val)
            accuracy_val_list.append(accuracy_val)
        dis_cost_val = sum(dis_cost_val_list) / len(dis_cost_val_list)
        dis_total_cost_val = sum(dis_total_cost_val_list) / len(dis_total_cost_val_list)
        accuracy_val = sum(accuracy_val_list) / len(accuracy_val_list)
        print("=========== iteration: ==========", iterations)
        print("test discriminator loss:", dis_cost_val)
        print("test discriminator total weigthted loss:", dis_total_cost_val)
        print("test discriminator accuracy:", accuracy_val)
        test_dis_cost_val_summary = tf.Summary(
            value=[tf.Summary.Value(tag="test_dis_cost", simple_value=dis_cost_val)])
        training_writer.add_summary(test_dis_cost_val_summary,
                                    tf.train.global_step(sess, global_step))
        test_dis_total_cost_val_summary = tf.Summary(
            value=[tf.Summary.Value(tag="test_dis_total_cost", simple_value=dis_total_cost_val)])
        training_writer.add_summary(test_dis_total_cost_val_summary,
                                    tf.train.global_step(sess, global_step))
        test_accuracy_val_summary = tf.Summary(
            value=[tf.Summary.Value(tag="test_accuracy", simple_value=accuracy_val)])
        training_writer.add_summary(test_accuracy_val_summary,
                                    tf.train.global_step(sess, global_step))
        # except KeyboardInterrupt


    with open(training_logs_dir + 'step' + str(iterations) + '_parameter.txt', 'w') as file:
        json.dump(conf, file)
        print("dumped conf info to " + training_logs_dir + 'step' + str(iterations) + '_parameter.txt')


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

    F_V_classification_conf = {
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
    validate_F_V_classification_fail(F_V_classification_conf)
