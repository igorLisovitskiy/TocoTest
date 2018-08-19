from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, os, shutil, time, tempfile, errno
import iris_data
import tensorflow as tf

import freeeze_graph
"""
freeze_graph
freeze_graph \
--input_graph=C:\Users\Lis\PycharmProjects\TensorFlowLiteConverter\model
--input_checkpoint=C:\Users\Lis\PycharmProjects\TensorFlowLiteConverter\model\model.ckpt-1000 \
--output_graph=/tmp/frozen_graph.pb \
--output_node_names=output_node \


"""


