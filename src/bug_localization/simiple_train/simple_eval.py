#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import csv
import json
import data_help


def eval_cnn(training_config_file=None):

    params = json.loads(open(training_config_file).read())

    # Load data
    print("Loading test data...")
    data_left, data_right, data_label, num_pos = \
        data_help.load_data_csv(params, os.path.join(params["data_dir"], params["test_bug_text"]),
                                   os.path.join(params["data_dir"], params["test_code_text"]),
                                   os.path.join(params["data_dir"], params["test_bug_code_index"]))

    # Map data into vocabulary
    left_vocab_path = os.path.join(params["checkpoint_dir"], "..", "left_vocab")
    print(left_vocab_path)
    left_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(left_vocab_path)
    x_left = np.array(list(left_vocab_processor.transform(data_left)))

    right_vocab_path = os.path.join(params["checkpoint_dir"], "..", "right_vocab")
    right_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(right_vocab_path)
    x_right = np.array(list(right_vocab_processor.transform(data_right)))

    print("Pos/Neg: {:d}/{:d}".format(num_pos, len(data_label) - num_pos))
    print("Pos Ratio: {:f}".format(float(num_pos) / float(len(data_label) - num_pos)))
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
            input_left = graph.get_operation_by_name("input_left").outputs[0]
            input_right = graph.get_operation_by_name("input_right").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Generate batches for one epoch
            batches = data_help.batch_iter(list(zip(x_left, x_right)), params["batch_size"], 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_scores = []

            for batch in batches:
                left_test_batch, right_test_batch = zip(*batch)
                batch_predictions = sess.run(predictions, {input_left: left_test_batch, input_right: right_test_batch, dropout_keep_prob: 1.0})
                batch_scores = sess.run(scores, {input_left: left_test_batch, input_right: right_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                # all_scores = np.concatenate([all_scores, batch_scores])

    # Print accuracy if y_test is defined
    if data_label is not None:
        # simple accuracy
        temp_predictions = all_predictions
        temp_data_label = data_label
        temp_num_pos = 0
        # simple accuracy
        correct_predictions = 0
        wrong_predictions = 0
        r_predictions = 0
        for i in range(len(temp_predictions)):
            if temp_data_label[i][1] == 1:
                temp_num_pos += 1
            if temp_data_label[i][1] == 1 and temp_predictions[i] == 1.0:
                correct_predictions += 1
            if temp_data_label[i][1] == 0 and temp_predictions[i] == 0.0:
                wrong_predictions += 1
            if (temp_data_label[i][1] == 0 and temp_predictions[i] == 0.0) \
                    or (temp_data_label[i][1] == 1 and temp_predictions[i] == 1.0):
                r_predictions += 1

        temp_num_neg = len(temp_data_label) - temp_num_pos
        temp_num_pos_percent = correct_predictions / float(temp_num_pos)
        temp_num_neg_percent = wrong_predictions / float(temp_num_neg)
        temp_mean_percent = (temp_num_pos_percent + temp_num_neg_percent) / 2
        print(
            "Total batch examples: {:d}, pos: {:d}, neg: {:d}".format(len(temp_data_label), temp_num_pos, temp_num_neg))

        print("Correct Accuracy: {:d}, {:d}, {:g}".format(correct_predictions, temp_num_pos, temp_num_pos_percent))
        print("Wrong Accuracy: {:d}, {:d}, {:g}".format(wrong_predictions, temp_num_neg, temp_num_neg_percent))
        print("Mean Accuracy: {:g}".format(temp_mean_percent))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((all_predictions, np.array(data_left), np.array(data_right)))
    out_path = os.path.join(params["checkpoint_dir"], "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)

    # predictions_human_readable = np.column_stack((np.array(data_left), np.array(data_right), all_scores))
    # out_path = os.path.join(params["checkpoint_dir"], "..", "prediction_scores.csv")
    # print("Saving evaluation to {0}".format(out_path))
    # with open(out_path, 'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)


if __name__ == '__main__':
    eval_cnn("./simple_training_config.json")