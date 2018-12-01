# coding=utf-8
import os
import time
import datetime
import math
import tensorflow as tf

from configs import train_config
from utils import util
from data_create import create_parse_tfr
from models import name_map_models
from trains.optimizer_tool import get_optimizer


class Trainer(object):
    def __init__(self, args):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.layers = None
        self.ave_vars = None

        if args is not None:
            self.batch_size = args.batch_size
            self.checkpoint_dir = args.ckpt_dir
            self.save_inter = args.save_inter
            self.test_inter = args.test_inter
            self.disp_inter = args.disp_inter
            self.show_detail = args.show_detail
            self.lr_type = args.lr_type
            if self.lr_type == 'naive':
                self.pre_loss = []
                self.cur_loss = []
                self.curr_lr_num = 0
                self.average_loss = 0.0
            if self.lr_type == 'sgdr':
                self.start_T = 0  # 每个周期的起点
                self.curr_T = train_config.T0  # 每个周期的终点

            if args.train_list_path is not None:
                self.train_list_path = args.train_list_path.split(',')
            else:
                self.train_list_path = None
            if args.test_list_path is not None:
                self.test_list_path = args.test_list_path.split(',')
            else:
                self.test_list_path = None

            self.need_padded_batch = args.need_padded_batch
            self.opt_type = args.opt_type
            self.show_detail_summary = args.show_detail_summary
            self.total_epochs = int(args.total_epochs)
            if self.total_epochs == 0:
                self.total_epochs = 1000000

            self.class_type = args.class_type
            try:
                self.char_to_label, self.label_to_char, self.class_num = util.get_class(self.class_type)
            except KeyError as e:
                tf.logging.info('class_type %s not found! you should register it in util/util.py-class_file_dict' %
                                self.class_type)
                raise e

            self.infer_model_path = args.infer_model_path
            self.gpu_id = args.gpu_id.split(',')
            self.model_type = args.model_type
            self.model = name_map_models.name2models_dict[args.model_type]
            self.print_config(args)

    def print_config(self, args):
        tf.logging.info('=' * 50)
        tf.logging.info('network config:')
        if args is not None:
            for arg in vars(args):
                tf.logging.info(str(arg) + ":" + str(getattr(args, arg)))
        # self.model.print_config()

    def build_for_train(self):
        tf.logging.info('=' * 30)
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tf.logging.info(str(localtime) + ' start to construct network...')

        with tf.variable_scope(tf.get_variable_scope()):
            global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0),
                                          trainable=False)  # step counter
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            # opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')
            opt = get_optimizer(self.opt_type, learning_rate, name='optimizer')
            # opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer')

            # create train and test dataset
            # buffer_size = util.num_class_set[self.class_type] * train_config.PRELOAD_BUFFER_RATIO
            buffer_size = self.class_num * train_config.PRELOAD_BUFFER_RATIO
            num_gpus = len(self.gpu_id)

            train_set = tf.data.TFRecordDataset(self.train_list_path)
            train_set = train_set.map(create_parse_tfr.dataset_parse_function)
            train_set = train_set.shuffle(buffer_size=buffer_size)
            train_set = train_set.repeat()
            if self.need_padded_batch:
                train_set = train_set.padded_batch(self.batch_size * num_gpus,
                                                   padded_shapes=([None, None, None], [None], [], []),
                                                   padding_values=(
                                                       tf.constant(0, dtype=tf.uint8), tf.constant(-1, dtype=tf.int64),
                                                       tf.constant(-1, dtype=tf.int32), ''))
            else:
                train_set = train_set.batch(self.batch_size * num_gpus)

            test_set = tf.data.TFRecordDataset(self.test_list_path)
            test_set = test_set.map(create_parse_tfr.dataset_parse_function)
            test_set = test_set.repeat(num_gpus)
            if self.need_padded_batch:
                test_set = test_set.padded_batch(self.batch_size * num_gpus,
                                                 padded_shapes=([None, None, None], [None], [], []),
                                                 padding_values=(
                                                     tf.constant(0, dtype=tf.uint8), tf.constant(-1, dtype=tf.int64),
                                                     tf.constant(-1, dtype=tf.int32), ''))
            else:
                test_set = test_set.batch(self.batch_size * num_gpus)

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_set.output_types, train_set.output_shapes)
            training_iterator = train_set.make_one_shot_iterator()
            validation_iterator = test_set.make_initializable_iterator()

            next_img, next_label, next_len, _ = iterator.get_next()
            image_splits = tf.split(next_img, num_gpus)
            label_splits = tf.split(next_label, num_gpus)
            len_splits = tf.split(next_len, num_gpus)

            tower_grads = []
            tower_loss = []
            tower_error_rate = []
            tower_prediction = []
            counter = 0
            for d in self.gpu_id:
                with tf.device('/gpu:%s' % d):
                    with tf.name_scope('%s_%s' % ('tower', d)):
                        prediction, error_rate, ctc_loss, _ = \
                            self.model.optimize_model(self.class_num, image_splits[counter],
                                                      label_splits[counter], len_splits[counter])
                        counter += 1
                        with tf.variable_scope("ctc"):
                            grads = opt.compute_gradients(ctc_loss)
                            tower_grads.append(grads)
                            tower_loss.append(ctc_loss)
                            tower_error_rate.append(error_rate)
                            tower_prediction.append(prediction)
                        tf.get_variable_scope().reuse_variables()

        mean_loss = tf.stack(axis=0, values=tower_loss)
        mean_loss = tf.reduce_mean(mean_loss, 0)
        mean_error_rate = tf.stack(axis=0, values=tower_error_rate)
        mean_error_rate = tf.reduce_mean(mean_error_rate, 0)
        mean_grads = util.average_gradients(tower_grads)
        clipped_grads = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in mean_grads]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # first 梯度更新
            apply_gradient_op = opt.apply_gradients(clipped_grads, global_step=global_step)
            # second 滑动平均
            variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            self.ave_vars = [variable_averages.average(var) for var in tf.trainable_variables()]
            train_op = tf.group(apply_gradient_op, variables_averages_op)
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tf.logging.info(str(localtime) + ' construct network finish')
        # save useful variables
        self.layers = {'global_step': global_step,
                       'learning_rate': learning_rate,
                       'optimizer': train_op,
                       'loss': mean_loss,
                       'error_rate': mean_error_rate,
                       'handle': handle,
                       'training_iterator': training_iterator,
                       'validation_iterator': validation_iterator
                       }
        self.train()

    def update_lr(self, curr_lr, step):
        if self.lr_type == 'sgdr':
            end_T = (step - self.start_T)
            if end_T == self.curr_T:
                self.start_T = step
                self.curr_T *= train_config.T_MUL
                return train_config.MAX_LR
            else:
                curr_lr = train_config.MIN_LR + \
                          0.5 * (train_config.MAX_LR - train_config.MIN_LR) * \
                          (1 + math.cos(math.pi * float(end_T) / self.curr_T))
                return curr_lr

        elif self.lr_type == 'naive':
            if step % self.test_inter == 0:
                if len(self.pre_loss) < train_config.LOSS_MOVING_AVERAGE_NUM or len(
                        self.cur_loss) < train_config.LOSS_MOVING_AVERAGE_NUM:
                    if len(self.pre_loss) < train_config.LOSS_MOVING_AVERAGE_NUM:
                        self.pre_loss.append(self.average_loss)
                    else:
                        self.cur_loss.append(self.average_loss)
                    return curr_lr
                else:
                    pre_mean = sum(self.pre_loss) / float(len(self.pre_loss))
                    cur_mean = sum(self.cur_loss) / float(len(self.cur_loss))
                    if (pre_mean - cur_mean) < train_config.END_CONDITION:
                        if self.curr_lr_num >= train_config.LEARNING_RATE_DECAY_NUM:
                            tf.logging.info('loss do not decrease, training finished!')
                            return -1.0
                        else:
                            tf.logging.info('loss do not decrease, learning rate decrease!')
                            curr_lr = curr_lr * train_config.LEARNING_RATE_DECAY_FACTOR
                            self.curr_lr_num += 1
                            del self.pre_loss[:]
                            del self.cur_loss[:]
                            return curr_lr
                    else:
                        self.cur_loss.append(self.average_loss)
                        first_loss = self.cur_loss.pop(0)
                        self.pre_loss.append(first_loss)
                        self.pre_loss.pop(0)
                        return curr_lr
            else:
                return curr_lr

    def train(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        localtime = time.asctime(time.localtime(time.time()))
        tf.logging.info(str(localtime) + ' start to train model...')
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        # tf_config.log_device_placement = True
        sess = tf.InteractiveSession(config=tf_config)
        # only save all trainable variables and global_step(for restore from previous training)
        save_vars = util.get_trainable_vars(self.layers['global_step']) + self.ave_vars
        saver = tf.train.Saver(var_list=save_vars, max_to_keep=20)

        # initialize and recover from latest model
        training_handle = sess.run(self.layers['training_iterator'].string_handle())
        validation_handle = sess.run(self.layers['validation_iterator'].string_handle())

        # initialize and recover from latest model
        sess.run(tf.global_variables_initializer())
        if tf.train.latest_checkpoint(self.checkpoint_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))

        # add some variables to summary
        loss_summary = tf.summary.scalar('loss', self.layers['loss'])
        error_summary = tf.summary.scalar('error_rate', self.layers['error_rate'])
        lr_summary = tf.summary.scalar('learning_rate', self.layers['learning_rate'])
        summaries = []
        if self.show_detail_summary:
            summaries = util.variable_summaries()
        summaries.extend([loss_summary, lr_summary, error_summary])
        train_summaries = tf.summary.merge(summaries)
        #train_summaries = tf.summary.merge(summaries.extend([loss_summary, lr_summary, error_summary]))

        train_file_writer = tf.summary.FileWriter(self.checkpoint_dir + '/train_log', sess.graph)
        test_file_writer = tf.summary.FileWriter(self.checkpoint_dir + '/test_log', sess.graph)

        if self.lr_type == 'naive':
            curr_lr = train_config.INITIAL_LEARNING_RATE
        elif self.lr_type == 'sgdr':
            curr_lr = train_config.MAX_LR
        else:
            raise Exception('wrong lr type!')

        sess.graph.finalize()  # 冻结图，防止训练中添加新的节点
        for epoch in range(self.total_epochs): # 这个地方有些问题，这样写step = total_epochs, 还有label的函数，最好再次重构一下结构
            start_time = time.time()
            _, step, loss, error_rate, lr_rate, train_merged = sess.run([self.layers['optimizer'],
                                                                         self.layers['global_step'],
                                                                         self.layers['loss'],
                                                                         self.layers['error_rate'],
                                                                         self.layers['learning_rate'],
                                                                         train_summaries],
                                                                        feed_dict={
                                                                            self.layers['handle']: training_handle,
                                                                            self.layers['learning_rate']: curr_lr})
            end_time = time.time()
            # save model
            if step % self.save_inter == 0:
                save_path = os.path.join(self.checkpoint_dir,
                                         self.class_type + '-' + self.model_type + '-model.ckpt')
                saver.save(sess, save_path, global_step=step)

            # save training info for displaying
            if step % self.disp_inter == 0:
                log_str = '[%s] %.3f sec/batch \t step:%d\t lr:%.6f \t loss:%.6f \t error rate:%.6f' % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (end_time - start_time), step, lr_rate, loss, error_rate)
                tf.logging.info(log_str)
                train_file_writer.add_summary(train_merged, step)

            # test model using test data
            if step % self.test_inter == 0:
                sess.run(self.layers['validation_iterator'].initializer)
                self.average_loss = 0.0
                average_error_rate = 0.0
                batch_num = 0
                while True:
                    try:
                        test_error_rate, test_loss = sess.run([
                            self.layers['error_rate'],
                            self.layers['loss']],
                            feed_dict={self.layers['handle']: validation_handle})
                        self.average_loss += test_loss
                        average_error_rate += test_error_rate
                        batch_num += 1
                    except tf.errors.OutOfRangeError:
                        break

                self.average_loss = self.average_loss / batch_num
                average_error_rate = average_error_rate / batch_num
                test_summaries = tf.Summary()
                loss_val = test_summaries.value.add()
                loss_val.tag = 'loss'
                loss_val.simple_value = self.average_loss
                acc_val = test_summaries.value.add()
                acc_val.tag = 'error_rate'
                acc_val.simple_value = average_error_rate
                test_file_writer.add_summary(test_summaries, step)

                localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tf.logging.info('=' * 30)
                log_str = str(localtime) + '\t test loss:%.6f \t test error rate:%.6f' % (
                    self.average_loss, average_error_rate)
                tf.logging.info(log_str)
                tf.logging.info('=' * 30)

            # update lr
            curr_lr = self.update_lr(curr_lr, step)
            if curr_lr < 0:
                break

    # just a sample for infer
    def build_and_infer(self):
        # create graph
        tf.logging.info('=' * 30)
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tf.logging.info(str(localtime) + ' start to construct network...')

        with tf.variable_scope(tf.get_variable_scope()):
            test_set = tf.data.TFRecordDataset(self.test_list_path)
            test_set = test_set.map(create_parse_tfr.dataset_parse_function)
            if self.need_padded_batch:
                test_set = test_set.padded_batch(self.batch_size, padded_shapes=([None, None, None], [None], [], []),
                                                 padding_values=(tf.constant(0, dtype=tf.uint8),
                                                                 tf.constant(-1, dtype=tf.int64),
                                                                 tf.constant(-1, dtype=tf.int32), ''))
            else:
                test_set = test_set.batch(self.batch_size)

            iterator = test_set.make_one_shot_iterator()
            next_img, next_label, next_len, next_filename = iterator.get_next()
            d = self.gpu_id[0]
            # with tf.device('/gpu:%s' % d):
            prediction, error_rate, ctc_loss, neg_sum_logits, y_pred = \
                self.model.build_model_tfr(self.class_num, next_img, next_label, next_len, self.batch_size,
                                           for_training=False)
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tf.logging.info(str(localtime) + ' construct network finish')
            # save useful variables
            self.layers = {'error_rate': error_rate,
                           'prediction': prediction,
                           'loss': ctc_loss,
                           'filename': next_filename,
                           'labels': next_label,
                           'neg_sum_logits': neg_sum_logits}

        self.infer(self.infer_model_path, self.batch_size)

    def infer(self, model_path, batch_size):
        tf.logging.info('start to infer')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.2
        sess = tf.InteractiveSession(config=tf_config)
        # only save all trainable varibales and global_step(for restore from previous training)
        # saver = tf.train.Saver(var_list=util.get_trainable_vars(self.layers['global_step']))
        variable_averages = tf.train.ExponentialMovingAverage(0.999)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, model_path)
        average_error_rate = 0.0
        total_loss = 0.0
        test_batch_num = 0
        seq_acc = 0.0
        start_time = time.time()
        # with codecs.open("wrong_preds.txt", 'w', 'utf-8') as w:
        img_path = '/Users/zhoukai/Desktop/refine_corp'
        wrong_img_path = '/Users/zhoukai/Desktop/wrong_img'
        while True:
            try:
                prediction, error_rate, loss, labels, filenames, neg_sum_logits = \
                    sess.run([self.layers['prediction'], self.layers['error_rate'],
                              self.layers['loss'], self.layers['labels'],
                              self.layers['filename'], self.layers['neg_sum_logits']])
                seq_acc += util.calc_batch_seq_acc(prediction, labels, filenames, self.show_detail, self.label_to_char,
                                                   neg_sum_logits)
                total_loss += loss
                average_error_rate += float(error_rate)
                test_batch_num += 1

            except tf.errors.OutOfRangeError:
                break
        end_time = time.time()
        tf.logging.info('prediction time:%f sec/batch' % ((end_time - start_time) / test_batch_num))
        average_error_rate = float(average_error_rate) / test_batch_num
        average_seq_acc = seq_acc / test_batch_num
        total_loss = float(total_loss) / test_batch_num
        tf.logging.info('testing  error rate:%f' % average_error_rate)
        tf.logging.info('testing loss:%f' % total_loss)
        tf.logging.info('average_seq_acc:%f' % average_seq_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_flag', type=str, default='train', help='train or infer mode')
    # some path
    parser.add_argument('--train_list_path', type=str, default=None, help='train tfr file path')
    parser.add_argument('--test_list_path', type=str, default=None, help='test tfr file path')
    parser.add_argument('--ckpt_dir', type=str, default='models', help='checkpoint file dir')
    parser.add_argument('--infer_model_path', type=str, default=None, help='model path under infer mode')
    # some data related variables
    parser.add_argument('--class_type', type=str, default='15', help='class set type:15,18,id,txt')
    parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
    # some training related variables
    parser.add_argument('--save_inter', type=int, default=10, help='save interval(batch num)')
    parser.add_argument('--test_inter', type=int, default=10, help='test interval(batch num)')
    parser.add_argument('--disp_inter', type=int, default=2, help='display in summary interval(batch num)')
    parser.add_argument('--show_detail', type=str, default='False', help='show infer detail for every image')
    parser.add_argument('--gpu_id', type=str, default='0', help='used gpu id,seperate by comma,ex:0,1,2')
    parser.add_argument('--model_type', type=str, default='desnetSeq', help='model type, current support two:crnn, cnn')
    parser.add_argument('--lr_type', type=str, default='sgdr', help='learning rate adjust strategy:naive,sgdr')

    parser.add_argument('--need_padded_batch', type=int, default=1, help='need padded batch:0,1')
    parser.add_argument('--opt_type', type=str, default='sgd', help='opt method')
    parser.add_argument('--show_detail_summary', type=int, default=1, help='show detail summary:0,1')
    parser.add_argument('--total_epochs', type=int, default=10, help='epochs')

    args, unparsed = parser.parse_known_args()

    trainer = Trainer(args)

    if args.mode_flag == 'train':
        trainer.build_for_train()
    elif args.mode_flag == 'infer':
        trainer.build_and_infer()
    else:
        raise Exception('wrong mode!')