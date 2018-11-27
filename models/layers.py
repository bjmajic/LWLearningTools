# coding=utf-8
"""
封装后TensorFlow 常见layer
"""

import tensorflow as tf


# **************** 卷积权重初始化 *********************
def xavier_initializer(uniform=True, seed=None):
    """
    如果激活函数使用sigmoid和tanh，怎最好使用xavir（someone think）
    :param uniform: if false, normal distributed random initialization
    :param seed:
    :return:
    """
    return tf.contrib.layers.xavier_initializer(uniform, seed)


def variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None):
    """
    如果使用relu，则最好使用he initial（someone think）
    各种factor, mode, uniform 的形式参考官方文档，这里给出的是配合selu使用的形式
    :param factor:
    :param mode:
    :param uniform:
    :param seed:
    :return:
    """
    return tf.contrib.layers.variance_scaling_initializer(factor, mode, uniform, seed)


def truncated_normal_initializer(stddev, mean=0.0):
    return tf.truncated_normal_initializer(mean=mean, stddev=stddev)


# ************** 正则化 *****************

def l2_regularizer(scale=0.0005):
    return tf.contrib.layers.l2_regularizer(scale)


def l1_regularizer(scale=0.0005):
    return tf.contrib.layers.l1_regularizer

# *******************************************


# ***************** 激活函数***************************************************
# someone 建议:
# Use ReLU with a samll learning rate
# Try out ELU / Leaky RELU / SELU
# Try out tanh but don not expect much and never use sigmoid
# Output layer should use softmax for classification or liner for regression
# *****************************************************************************

def relu(features, name=None):
    """
    :param features:
    :param name:
    :return: max(features, 0)
    """
    return tf.nn.relu(features=features, name=name)


def relu6(features, name=None):
    """
    min(max(features, 0), 6)
    :param features:
    :param name:
    :return: min(max(features, 0), 6)
    """
    return tf.nn.relu6(features=features, name=name)


def prelu(features, name=None):
    pass


def crelu(features, axis=-1, name=None):
    """
    # concat 后的结果为：[relu(features), -relu(features)]，一个是relu，一个是relu关于y轴对称的形状
    :param features:
    :param axis:
    :param name:
    :return:
    """
    return tf.nn.crelu(features=features, name=name)


def selu(features, name=None):
    """
    Computes scaled exponential linear: scale * alpha * (exp(features) - 1)
    if < 0, scale * features otherwise.
    To be used together with initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN').
    For correct dropout, use tf.contrib.nn.alpha_dropout.
    some one think: 经过该激活函数后使得样本分布自动归一化到0均值和单位方差(自归一化，保证训练过程中梯度不会爆炸或消失，
    效果比Batch Normalization 要好)
    :param features:
    :param name:
    :return:
    """
    return tf.nn.selu(features=features, name=name)

# ********************************************************************


def conv2d(inputs, filters, kernel_size, strides, padding,
           dilation_rate=(1, 1), activation=selu, use_bias=True,
           kernel_initializer=variance_scaling_initializer(), kernel_regularizer=None, bias_regularizer=None,
           trainable=True, name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding,
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            trainable=trainable, name=name)


def max_pool2d(inputs, pool_size, strides, padding='valid', name=None):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding, name=name)


def ave_pool2d(inputs, pool_size, strides, padding='valid', name=None):
    return tf.layers.average_pooling2d(inputs, pool_size, strides, padding, name=name)


def global_ave_pool2d(inputs, stride=1, data_format='NHWC'):
    input_shape = inputs.get_shape().as_list()
    if data_format == 'NHWC':
        width = input_shape[2]
        height = input_shape[1]
    elif data_format == 'NCHW':
        width = input_shape[3]
        height = input_shape[2]
    else:
        width = 1
        height = 1
        print("invalid data format")
    return tf.layers.average_pooling2d(inputs=inputs, pool_size=(height, width), strides=stride)


def flatten(inputs, name=None):
    return tf.layers.flatten(inputs, name)


def fully_connected(inputs, num_outputs, activation_fn=tf.nn.softmax, use_bias=True,
                    kernel_initializer=xavier_initializer(),
                    kernel_regularizer=l2_regularizer(), bias_regularizer=None,
                    trainable=True, name=None):
    inputs_flatten = tf.layers.flatten(inputs)
    return tf.layers.dense(inputs_flatten, num_outputs, activation=activation_fn, use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                           trainable=trainable, name=name)


def dense(inputs, num_outputs, activation_fn=tf.nn.softmax, use_bias=True, kernel_initializer=xavier_initializer(),
          kernel_regularizer=l2_regularizer(), bias_regularizer=None,
          trainable=True, name=None):
    return tf.layers.dense(inputs, num_outputs, activation=activation_fn, use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                           trainable=trainable, name=name)


def batch_normalization(inputs, training=True, name=None):
    return tf.layers.batch_normalization(inputs, training=training, name=name)


def drop_out(inputs, rate, training=True, name=None):
    return tf.layers.dropout(inputs, rate=rate, training=training, name=name)


def concatenation(layer_list, axis=3):
    """
    按照指定的axis拼接layer
    :param layer_list: layer的列表
    :param axis: 拼接轴
    :return:
    """
    return tf.concat(layer_list, axis=axis)


def time_distributed(inputs, class_num, time_axis='w', name=None):
    # **************************
    # 关于使用Keras中的TimeDistributed，使用tensorflow,先reshape 一下，保留时间轴，然后使用dense执行分类（dense的输出结果和
    # 输入的维度一致（除了最后一维）
    #  ***************************
    if time_axis == 'w':
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
    # reshape
    feature_shape = inputs.get_shape().as_list()
    # print feature_shape[2]
    dims = feature_shape[2] * feature_shape[3]
    feature_map_reshaped = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], dims])
    outputs = tf.layers.dense(feature_map_reshaped, class_num, name=name)
    return outputs
