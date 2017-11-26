from scipy import misc
import glob
import os.path
import numpy as np
import tensorflow as tf
import cv2
from util import *

def save_data2record(tf_path, file_path):
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    label2int = 0
    for class_path in glob.glob(file_path):
        for image_path in glob.glob(class_path + "/*.jpg"):
            image = scipy.misc.imread(image_path)
            if len(image.shape) != 3:
                continue
            image_raw = crop2Target(image).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label2int])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
            writer.write(example.SerializeToString())
        label2int += 1
    writer.close()


def read_and_decode(file_name):
    fileName_queue = tf.train.string_input_producer([file_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileName_queue)
    features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'image_raw': tf.FixedLenFeature([], tf.string),
                                        })
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [96, 96, 3])
    img = normalizaion(tf.cast(img, tf.float32))
    label = tf.cast(features['label'], tf.int32)

    return img, label

def tensor_decode():
    img, label = read_and_decode("train.tfrecords")

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=70, capacity=200,
                                                    min_after_dequeue=100)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            print(val, l)

if __name__ == '__main__':
    # save_data2record('train.tfrecords', '../data/image_sample/*')
    tensor_decode()