# coding=utf-8
import create_parse_tfr
import tensorflow as tf
import os
import random
import codecs
import cv2


def create_txt():
    base_root = r'/Users/raymond/Downloads/cifar-10'
    train_root = os.path.join(base_root, 'train')
    test_root = os.path.join(base_root, 'test')

    image_list = []
    for i in range(10):
        sub_train_folder = os.path.join(train_root, str(i))
        images = os.listdir(sub_train_folder)
        for image in images:
            if image.endswith('.jpg'):
                image_list.append('%s\t%s\n' % (os.path.join(sub_train_folder, image), str(i)))

    random.shuffle(image_list)
    with open('train.txt', 'w') as f:
        f.writelines(image_list)

    image_list = []
    for i in range(10):
        sub_test_folder = os.path.join(test_root, str(i))
        images = os.listdir(sub_test_folder)
        for image in images:
            if image.endswith('.jpg'):
                image_list.append('%s\t%s\n' % (os.path.join(sub_test_folder, image), str(i)))

    random.shuffle(image_list)
    with open('test.txt', 'w') as f:
        f.writelines(image_list)


def create_tfr():
    # char_to_label, label_to_char, class_num = util.get_class('name')
    list_path1 = 'test.txt'
    save_path = 'test'
    split_num = 1
    samples = codecs.open(list_path1, 'r', 'utf-8').readlines()
    random.shuffle(samples)

    part_num = len(samples) / split_num
    for idx in range(split_num):
        writer = tf.python_io.TFRecordWriter(save_path + str(idx) + '.tfr')
        split_samples = samples[part_num * idx:part_num * (idx + 1)]
        counter = 0
        print 'gen tfr index:', idx
        for s in split_samples:
            info = s.split('\t')

            filename = info[0].strip()
            classes = int(info[1].strip())

            img = cv2.imread(filename)
            if img is None:
                print('none' + filename)

            ret, str_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded_jpg = str_img.tostring()
            # with tf.gfile.GFile(img_path, 'rb') as fid:
            #     encoded_jpg = fid.read()

            features = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),
                        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')]))}
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            counter += 1
            if counter % 100 == 0:
                print counter
        writer.close()


def dataset_parse_function(sample_proto):
    features = {'image': tf.FixedLenFeature((), tf.string, default_value=""),  # 是一个标量，string
                'label': tf.FixedLenFeature((), tf.int64),
                'filename': tf.FixedLenFeature((), tf.string, default_value="")}

    parsed_features = tf.parse_single_example(sample_proto, features)
    decoded_image = tf.image.decode_jpeg(parsed_features['image'])
    # decoded_image = tf.image.decode_bmp(parsed_features['image'])
    dense_label = tf.cast(parsed_features['label'], tf.int32)
    # widths = tf.cast(parsed_features['width'], tf.int32)
    return decoded_image, dense_label, parsed_features['filename']


if __name__ == '__main__':
    create_tfr()
    # create_txt()
