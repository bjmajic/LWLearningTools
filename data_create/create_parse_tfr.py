# coding=utf-8

import tensorflow as tf
import codecs
import random
import os
import cv2
import math


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dense_to_sparse(dense_tensor, out_type, constant_value=-1):
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(constant_value, dense_tensor.dtype)))
    values = tf.cast(tf.gather_nd(dense_tensor, indices), tf.int32)
    shape = tf.shape(dense_tensor, out_type=out_type)
    return tf.SparseTensor(indices, values, shape)


# ********* just a example ******************
def get_class(typestr):
    return 1, 2, 3
# *******************************************


def gen_bl_tfr():
    char_to_label, label_to_char, class_num = get_class('name')
    list_path = '/data/songqi03/projects/waimai_data_ocr_data_synthesis/output/addr_20180529_res.txt'
    folder_path = '/data/songqi03/projects/waimai_data_ocr_data_synthesis/output/addr_20180529_res'
    save_path = '/data/songqi03/projects/waimai_data_ocr_data_synthesis/output/addr_20180529_res'
    split_num = 1
    samples = codecs.open(list_path, 'r', 'utf-8').readlines()
    random.shuffle(samples)
    # samples=samples[:40000]
    part_num = len(samples) / split_num
    for idx in range(split_num):
        writer = tf.python_io.TFRecordWriter(save_path + str(idx) + '.tfr')
        split_samples = samples[part_num * idx:part_num * (idx + 1)]
        counter = 0
        print 'gen tfr index:', idx
        for s in split_samples:
            info = s.split('\t')
            filename = info[0].strip()
            classes_text = info[1].strip()

            # info = eval(s)
            # filename = info['picture_name'].strip()
            # classes_text = info['string'].strip()

            classes = []
            for t in classes_text:
                classes.append(char_to_label[t])
            # print filename
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            # if resize_flag:
            #     img = util.resize_img(img, h)
            img_width = img.shape[1]
            if int(math.ceil(img_width / 4)) - 1 < len(classes):
                print filename, 'too short!'
                continue
            ret, str_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded_jpg = str_img.tostring()

            # with tf.gfile.GFile(img_path, 'rb') as fid:
            #     encoded_jpg = fid.read()

            features = {'image': _bytes_feature(encoded_jpg),
                        'label': _int64_feature(classes),
                        'width': _int64_feature(img_width),
                        'filename': _bytes_feature(filename.encode('utf8'))
                        }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            counter += 1
            if counter % 100 == 0:
                print counter
        writer.close()


def dataset_parse_function(sample_proto):
    features = {'image': tf.FixedLenFeature((), tf.string, default_value=""),  # 是一个标量，string
                'label': tf.VarLenFeature(tf.int64),  # 因为这个label是一个向量，并且各个向量的长度不等，所以使用VarLenFeature
                'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                'filename': tf.FixedLenFeature((), tf.string, default_value="")}

    parsed_features = tf.parse_single_example(sample_proto, features)
    decoded_image = tf.image.decode_jpeg(parsed_features['image'])
    # decoded_image = tf.image.decode_bmp(parsed_features['image'])
    dense_label = tf.sparse_tensor_to_dense(parsed_features['label'], default_value=-1)
    widths = tf.cast(parsed_features['width'], tf.int32)
    return decoded_image, dense_label, widths, parsed_features['filename']
