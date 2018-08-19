#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,                         software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from  tensorflow.contrib.lite.python import lite

import argparse
import tensorflow as tf


import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y,
                                                 args.batch_size))

    print('\nTest set accuracy:         {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))

    print("\n====== classifier model_dir, latest_checkpoint ===========")
    print(classifier.model_dir)
    print(classifier.latest_checkpoint())
    debug = False

    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        latest_checkpoint_path = classifier.latest_checkpoint()
        saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
        saver.restore(sess, latest_checkpoint_path)

        # Get the input and output tensors needed for toco.
        # These were determined based on the debugging info printed / saved below.
        input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
        input_tensor.set_shape([1, 4])
        out_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")
        out_tensor.set_shape([1, 3])

        # Pass the output node name we are interested in.
        # Based on the debugging info printed / saved below, pulled out the
        # name of the node for the logits (before the softmax is applied).
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=["dnn/logits/BiasAdd"])

        if debug is True:
            print("\nORIGINAL GRAPH DEF Ops ===========================================")
            ops = sess.graph.get_operations()
            for op in ops:
                if "BiasAdd" in op.name or "input_layer" in op.name:
                    print([op.name, op.values()])
            # save original graphdef to text file
            with open("estimator_graph.pbtxt", "w") as fp:
                fp.write(str(sess.graph_def))
            print("\nFROZEN GRAPH DEF Nodes ===========================================")
            for node in frozen_graph_def.node:
                print(node.name)
            # save frozen graph def to text file
            with open("estimator_frozen_graph.pbtxt", "w") as fp:
                fp.write(str(frozen_graph_def))

    tflite_model = lite.toco_convert(frozen_graph_def, [input_tensor], [out_tensor])
    open("estimator_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)