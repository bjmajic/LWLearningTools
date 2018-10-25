# coding=utf-8
import tensorflow as tf


def get_optimizer(type_name, learning_rate, name='optimizer'):
    if type_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name=name)
    elif type_name == 'moment':
        return tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif type_name == 'Adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name=name)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name=name)