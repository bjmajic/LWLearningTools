# coding=utf-8
import crnn_model
import densnet_model
import densenet_seq_model
from configs import densenet_config


def densenet():
    return densnet_model.Net(densenet_config.filters, densenet_config.training,
                             densenet_config.class_num, densenet_config.dropout_rate)


def densenet_seq():
    return densenet_seq_model.Net(densenet_config.filters, densenet_config.training,
                                  densenet_config.class_num, densenet_config.dropout_rate)


name2models_dict = {
    'crnn': crnn_model,
    'desnet': densenet(),
    'desnetSeq': densenet_seq()
}