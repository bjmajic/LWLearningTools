# -*- coding: UTF-8 -*-

# input image setting
img_mean = 128.0
img_var = 128.0
img_height = 32
img_width = 32
img_ch = 3

# tfr pipline setting
PRELOAD_BUFFER_RATIO = 100

# OHEM ratio
OHEM_NUM_RATIO = 1.0

# naive learning rate adjust settings
INITIAL_LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.5
LOSS_MOVING_AVERAGE_NUM = 10
END_CONDITION = 0.001
LEARNING_RATE_DECAY_NUM = 3
DECAY_STEPS = 200000

# SGDR learning rate adjust settings
MIN_LR = 0.0  # max learning rate
MAX_LR = 0.2  # min learning rate
T0 = 20000  # initial peroid (just how many step)
T_MUL = 2  # multi factor for every peroid


# which model being for train
MODEL_NAME = 'desnetSeq'
class_num = 10
