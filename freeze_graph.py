# -*- coding: UTF-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts checkpoint variables into Const ops in a standalone GraphDef file.

This script is designed to take a GraphDef proto, a SaverDef proto, and a set of
variable values stored in a checkpoint file, and output a GraphDef with all of
the variable ops converted into const ops containing the values of the
variables.

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded where we may not have access to the
RestoreTensor ops and file loading calls that they rely on.
"""

import argparse
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import numpy as np
from model.cnn_model import CNNModel
from model.crnn_model import CRNNModel
from utils import util


FLAGS = None

"""
--input_graph=/Users/songqi/meituan_workspace/tmp/data_refine_20_drop05/models/crnn-model.ckpt-316000.meta
--input_graph=models/crnn-model.ckpt-200.meta
--input_checkpoint=models/crnn-model.ckpt-200
--output_graph=models/frozen_models200.pb2
--output_node_names=prediction
"""

model_dict = {'crnn': CRNNModel(), 'cnn': CNNModel()}


def remove_training_nodes(input_graph):
    """Prunes out nodes that aren't needed for inference.

    There are nodes like Identity and CheckNumerics that are only useful
    during training, and can be removed in graphs that will be used for
    nothing but inference. Here we identify and remove them, returning an
    equivalent graph. To be specific, CheckNumerics nodes are always removed, and
    Identity nodes that aren't involved in control edges are spliced out so that
    their input and outputs are directly connected.

    Args:
      input_graph: Model to analyze and prune.

    Returns:
      A list of nodes with the unnecessary ones removed.
    """

    types_to_remove = {"CheckNumerics": True}

    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove:
            names_to_remove[node.name] = True

    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)

    types_to_splice = {"Identity": True}
    names_to_splice = {}
    for node in nodes_after_removal:
        if node.op in types_to_splice:
            # We don't want to remove nodes that have control edge inputs, because
            # they might be involved in subtle dependency issues that removing them
            # will jeopardize.
            has_control_edge = False
            for input_name in node.input:
                # add by songqi
                if re.match(r"^\^", input_name) or re.match(r".*/while/.*", input_name):
                    has_control_edge = True
            if not has_control_edge:
                names_to_splice[node.name] = node.input[0]

    nodes_after_splicing = []
    for node in nodes_after_removal:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub(r"^\^", "", full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    output_graph.library.CopyFrom(input_graph.library)
    output_graph.versions.CopyFrom(input_graph.versions)
    return output_graph


def freeze_crnn_graph(args):
    with tf.Session() as sess:
        if args.input_graph is not None:
            # 不建议使用这个方法，因为ckpt保存下来图模型，所有带有is_training选项的都是True
            saver = tf.train.import_meta_graph(args.input_graph, clear_devices=True)
        else:
            model = model_dict[args.model_type]
            model.print_config()
            char_to_label, label_to_char_maps, class_num = util.get_class(args.seq_type)
            model.build_model(class_num, for_training=False)
            # var_list = util.get_trainable_vars(None)
            # saver = tf.train.Saver(var_list=var_list)
            variable_averages = tf.train.ExponentialMovingAverage(0.999)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

        saver.restore(sess, args.input_checkpoint)
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        compressed_graph_def = remove_training_nodes(graph_def)
        const_graph_def = graph_util.convert_variables_to_constants(sess, compressed_graph_def, ['ctc/prediction'])

        output_graph_def = graph_pb2.GraphDef()
        for input_node in const_graph_def.node:
            output_node = node_def_pb2.NodeDef()
            if input_node.name == 'input/is_training':
                output_node.op = "Const"
                output_node.name = input_node.name
                dtype = input_node.attr["dtype"]
                data = np.array(False, dtype=np.bool)
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue
                                                   (tensor=tensor_util.make_tensor_proto(data,
                                                                                         dtype=dtype.type,
                                                                                         shape=data.shape)))
            else:
                output_node.CopyFrom(input_node)
            output_node.device = ""
            output_graph_def.node.extend([output_node])
        output_graph_def.library.CopyFrom(const_graph_def.library)

        with gfile.GFile(args.output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print "%d ops in the final graph." % len(output_graph_def.node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_graph",
        type=str,
        default=None,
        help="TensorFlow \'GraphDef\' file to load.")
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        default="",
        help="TensorFlow variables file to load.")
    parser.add_argument(
        "--output_graph",
        type=str,
        default="",
        help="Output \'GraphDef\' file name.")
    parser.add_argument(
        "--seq_type",
        type=str,
        default="15",
        help="seq type:15,18,id or txt")
    parser.add_argument(
        "--model_type",
        type=str,
        default="crnn",
        help="model type:cnn or crnn")
    args = parser.parse_args()
    freeze_crnn_graph(args)
