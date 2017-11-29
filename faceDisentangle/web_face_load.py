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


def read_and_decode(file_name, num_epochs=1):
    fileName_queue = tf.train.string_input_producer([file_name], num_epochs=num_epochs, shuffle=True)

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

def read_and_decode_uint8(file_name):
    fileName = tf.train.string_input_producer([file_name], num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileName)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [96, 96, 3])
    label = tf.cast(features['label'], tf.int32)

    return img, label


def tensor_decode(batch_size, n_epochs):
    img, label = read_and_decode("train.tfrecords", n_epochs)

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=200,
                                                    min_after_dequeue=100,
                                                    allow_smaller_final_batch=True)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                val, l = sess.run([img_batch, label_batch])
                print l
        except tf.errors.OutOfRangeError:
            print('Done data collecting, epoch reached')
        finally:
            coord.request_stop()
        coord.join(threads)


def tensor_decode_all(filePath):
    record_iterator = tf.python_io.tf_record_iterator(path=filePath)

    img_list = []
    label_list = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        label = (example.features.feature['label']
                      .int64_list
                      .value[0])

        img_raw = (example.features.feature['image_raw']
                      .bytes_list
                      .value[0])

        img_raw = np.fromstring(img_raw, dtype=np.uint8)
        img = img_raw.reshape((96, 96, 3))
        img_list.append(img)
        label_list.append(int(label))

    return np.array(img_list), np.array(label_list)


if __name__ == '__main__':
    # save_data2record('train.tfrecords', '../data/image_sample/*')
    # tensor_decode(8, 2)
    img_list, label_list = tensor_decode_all('train.tfrecords')
    img_list.astype(dtype=np.float32)
    print img_list, label_list