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
            x = layers.batch_normalization(x, training=self.training, name=scope + '_batch1')
            x = layers.selu(x)
            x = layers.conv2d(x, filters=4 * self.filters, kernel_size=[1, 1], strides=[1, 1],
                              kernel_regularizer=layers.l2_regularizer(0.0005),
                              padding='same', activation=None, name=scope + '_conv1')
            x = layers.drop_out(x, rate=self.dropout, training=self.training)

            x = layers.batch_normalization(x, training=self.training, name=scope + '_batch2')
            x = layers.selu(x)
            x = layers.conv2d(x, filters=self.filters, kernel_size=[3, 3], strides=[1, 1],
                              kernel_regularizer=layers.l2_regularizer(0.0005),
                              padding='same', activation=None, name=scope + '_conv2')
            x = layers.drop_out(x, rate=self.dropout, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.variable_scope(scope):
            x = layers.batch_normalization(x, training=self.training, name=scope + '_batch1')
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
        x = layers.conv2d(input_x, filters=2*self.filters, kernel_size=[7, 7], strides=[2, 2],
                          kernel_regularizer=layers.l2_regularizer(0.0005),
                          padding='valid', activation=None, name='conv0')

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = layers.batch_normalization(x, training=self.training, name='linear_batch')
        x = layers.selu(x)
        x = layers.global_ave_pool2d(x)
        # x = flatten(x)
        x = layers.fully_connected(x, self.class_num, use_bias=False, activation_fn=None, trainable=self.training,
                                   name='full_connecting')

        # x = tf.reshape(x, [-1, 10])
        return x

    def optimize_model(self, input_x, input_y):
        logits = self.Dense_net(input_x)
        labels = tf.reshape(input_y, shape=[-1])
        labels = tf.one_hot(labels, self.class_num)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return cost, accuracy


if __name__ == '__main__':
    model = Net(24, True, 1000, 0.2)
    input_x = tf.ones([1, 32, 320, 3], dtype=np.float32)
    res = model.Dense_net(input_x)
    print(res)

