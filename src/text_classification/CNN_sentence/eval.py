#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import json

def eval_cnn(training_config_file=None):

    params = json.loads(open(training_config_file).read())

    # CHANGE THIS: Load data. Load your own data here
    if params["eval_train"]:
        x_raw, y = data_helpers.load_text_and_path_label(params["file_level_train_file"], params["file_level_label"])
        y_test = np.argmax(y, axis=1)
    else:
        x_raw, y = data_helpers.load_text_and_path_label(params["file_level_test_file"], params["file_level_label"])
        y_test = np.argmax(y, axis=1)

    # Map data into vocabulary
    vocab_path = os.path.join(params["checkpoint_dir"], "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(params["checkpoint_dir"])
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=params["allow_soft_placement"],
          log_device_placement=params["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), params["batch_size"], 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = 0
        for i in range(len(all_predictions)):
            if y_test[i] == 1 and all_predictions[i] == 1.0:
                correct_predictions += 1
        # correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(params["checkpoint_dir"], "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)


if __name__ == '__main__':
    eval_cnn("./training_config")