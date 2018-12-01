# coding=utf-8
import tensorflow as tf


def get_class(type_name):
    labels = {}  # 标签的数字表示
    labels_depict = {}  # 标签的描述，汉字or英文描述
    class_num = 100  # 类别总数
    return labels, labels_depict, class_num


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_trainable_vars(global_step):
    var_list = tf.trainable_variables()
    if global_step is not None:
        var_list.append(global_step)
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    return var_list


def variable_summaries():
    var_list = tf.trainable_variables()
    ret_summary = []
    with tf.name_scope('summaries'):
        for var in var_list:
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            # tf.summary.scalar('mean', mean)
            ret_summary.append(tf.summary.scalar('mean', mean))
            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            ret_summary.append(tf.summary.scalar('stddev', stddev))
            ret_summary.append(tf.summary.scalar('max', tf.reduce_max(var)))
            ret_summary.append(tf.summary.scalar('min', tf.reduce_min(var)))

            # 用直方图记录参数的分布
            ret_summary.append(tf.summary.histogram('histogram', var))
    return ret_summary
