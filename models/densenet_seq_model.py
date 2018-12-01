# coding=utf-8
import layers
import tensorflow as tf
import numpy as np


class Net(object):
    def __init__(self, filters, training, class_num, dropout_rate):  #
        self.filters = filters
        self.training = training
        # self.model = self.Dense_net(x)
        self.model = None
        self.dropout = dropout_rate
        self.class_num = class_num

        self.inputs = []
        self.outputs = []

        # self.x = x
        # self.y = labels

    def bottleneck_layer(self, x, scope):
        with tf.variable_scope(scope):
            # x = layers.batch_normalization(x, training=self.training, name=scope + '_batch1')
            x = layers.selu(x)
            x = layers.conv2d(x, filters=4 * self.filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_regularizer=layers.l2_regularizer(0.0005),
                              padding='same', activation=None, name=scope + '_conv1')
            x = layers.drop_out(x, rate=self.dropout, training=self.training)

            # x = layers.batch_normalization(x, training=self.training, name=scope + '_batch2')
            x = layers.selu(x)
            x = layers.conv2d(x, filters=self.filters, kernel_size=[3, 3], strides=[1, 1],
                              kernel_regularizer=layers.l2_regularizer(0.0005),
                              padding='same', activation=None, name=scope + '_conv2')
            x = layers.drop_out(x, rate=self.dropout, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.variable_scope(scope):
            # x = layers.batch_normalization(x, training=self.training, name=scope + '_batch1')
            x = layers.selu(x)
            x = layers.conv2d(x, filters=self.filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_regularizer=layers.l2_regularizer(0.0005),
                              padding='same', activation=None, name=scope + '_conv1')
            x = layers.drop_out(x, rate=self.dropout, training=self.training)
            x = layers.ave_pool2d(x, pool_size=[2, 2], strides=[2, 2])
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.variable_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = layers.concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = layers.concatenation(layers_concat)
            return x

    def Dense_net(self, input_x):
        x = layers.conv2d(input_x, filters=2*self.filters, kernel_size=[5, 5], strides=[2, 2],
                          kernel_regularizer=layers.l2_regularizer(0.0005),
                          padding='valid', activation=None, name='conv0')

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')

        # x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
        x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_final')

        # 100 Layer
        # x = layers.batch_normalization(x, training=self.training, name='linear_batch')
        x = layers.selu(x)
        # x = layers.global_ave_pool2d(x)
        # x = flatten(x)
        # x = layers.fully_connected(x, self.class_num, use_bias=False, activation_fn=None, trainable=self.training,
                                   # name='full_connecting')

        # x = tf.reshape(x, [-1, 10])
        return x

    def dense_to_sparse(self, dense_tensor, out_type):
        indices = tf.where(tf.not_equal(dense_tensor, tf.constant(-1, dense_tensor.dtype)))
        values = tf.cast(tf.gather_nd(dense_tensor, indices), tf.int32)
        shape = tf.shape(dense_tensor, out_type=out_type)
        return tf.SparseTensor(indices, values, shape)

    def rnn_model(self, cnn_features, labels, seq_len, class_num):
        # W 作为时间轴维度
        rnn_input = tf.transpose(cnn_features, [0, 2, 1, 3])
        # reshape
        feature_shape = rnn_input.get_shape().as_list()
        # print feature_shape[2]
        dims = feature_shape[2] * feature_shape[3]
        rnn_input = tf.reshape(rnn_input, [tf.shape(rnn_input)[0], tf.shape(rnn_input)[1], dims], name='rnn_input')
        sparse_label = self.dense_to_sparse(labels, tf.int64)
        real_len = tf.add(tf.floor(tf.cast(seq_len, dtype=tf.float32) / 8), -1)
        real_len = tf.cast(real_len, dtype=tf.int32, name='real_len')

        lstm_fw_cells = [
            tf.contrib.rnn.LSTMCell(512, use_peepholes=False, forget_bias=1.0) for _ in range(2)]
        lstm_bw_cells = [
            tf.contrib.rnn.LSTMCell(512, use_peepholes=False, forget_bias=1.0) for _ in range(2)]
        lstm_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells,
                                                                           lstm_bw_cells,
                                                                           rnn_input,
                                                                           dtype=tf.float32,
                                                                           sequence_length=real_len,
                                                                           parallel_iterations=256)
        # expand_output = tf.expand_dims(lstm_output, 1, name='expand_output')
        # 生成每个time 的所有分类，注意不要使用softmax，因为ctc内部会做
        class_conv = layers.dense(lstm_output, class_num, None, trainable=self.training, name='ctc_input')
        return class_conv, sparse_label, real_len

    def ctc_model(self, sparse_label, ctc_input, real_len):
        ctc_input = tf.transpose(ctc_input, [1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(sparse_label, ctc_input, real_len, ignore_longer_outputs_than_inputs=False,
                                  time_major=True)
        ctc_loss = tf.reduce_mean(ctc_loss, name='ctc_loss')
        sparse_preds, neg_sum_logits = tf.nn.ctc_greedy_decoder(ctc_input, real_len, merge_repeated=True)
        error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(sparse_preds[0], tf.int32), sparse_label),
                                    name='error_rate')
        prediction = []
        if not self.training:
            prediction = tf.sparse_tensor_to_dense(sparse_preds[0], default_value=-1, name='prediction')
        return prediction, error_rate, ctc_loss, neg_sum_logits

    def optimize_model(self, class_num, input_x, labels, seq_len, img_height=32, img_ch=3):
        image = tf.reshape(input_x, [tf.shape(input_x)[0], img_height, -1, img_ch], 'image')
        float_image = tf.cast(image, tf.float32, name='float_image')
        if self.training:
            float_image = tf.image.random_contrast(float_image, 0.5, 1.5)
            float_image = tf.image.random_brightness(float_image, 0.3)
            float_image = tf.image.random_hue(float_image, 0.5)
            float_image = tf.image.random_saturation(float_image, 0.0, 2.0)
        norm_image = (float_image - 127.5) / 127.5
        cnn_features = self.Dense_net(norm_image)

        with tf.variable_scope('rnn'):
            ctc_input, sparse_label, real_len = self.rnn_model(cnn_features, labels, seq_len, class_num)
            # return ctc_input, sparse_label, real_len

        with tf.variable_scope("ctc"):
            prediction, error_rate, ctc_loss, neg_sum_logits = self.ctc_model(sparse_label, ctc_input, real_len)
            return prediction, error_rate, ctc_loss, neg_sum_logits

    def frozen_model(self, class_num):
        self.training = False
        with tf.variable_scope("input"):
            image = tf.placeholder(dtype=tf.uint8, shape=(None, 32, None, 3),
                                   name='image')
            float_image = tf.cast(image, tf.float32, name='float_image')
            norm_image = (float_image - 127.5) / 127.5
            # sparse represent of label fot ctc loss
            label_id = tf.placeholder(dtype=tf.int64, name='label_id')
            label_value = tf.placeholder(dtype=tf.int32, name='label_value')
            label_shape = tf.placeholder(dtype=tf.int64, name='label_shape')
            seq_len = tf.placeholder(tf.int32, shape=(image.shape[0]), name='seq_len')
        feed_dict = {'image': image, 'label_id': label_id, 'label_value': label_value, 'label_shape': label_shape,
                     'seq_len': seq_len}
        cnn_features = self.Dense_net(norm_image)

        with tf.variable_scope('rnn'):
            ctc_input, sparse_label, real_len = self.rnn_model(cnn_features, label_value, seq_len, class_num)
            # return ctc_input, sparse_label, real_len

        with tf.variable_scope("ctc"):
            self.ctc_model(sparse_label, ctc_input, real_len)


if __name__ == '__main__':
    model = Net(24, True, 1000, 0.2)
    input_x = tf.ones([1, 32, 320, 3], dtype=np.float32)
    #res = model.Dense_net(input_x)
    #print(res)
    res2 = model.optimize_model(1000, input_x, tf.constant([[1, 2, 3, -1, -1]]), [320])
    print(res2)

