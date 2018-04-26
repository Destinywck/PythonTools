#!/usr/bin/env python
# encoding=utf-8

import os
import sys
import time
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from bug_code_cnn import BugCodeCNN
import data_help


def main(training_config_file):
    params = json.loads(open(training_config_file).read())
    # Load data
    print("Loading train data...")
    data_left, data_right, data_label, num_pos = \
        data_help.load_data_csv(params, os.path.join(sys.path[0], "data", params["train_bug_text"]),
                  os.path.join(sys.path[0], "data", params["train_code_text"]),
                  os.path.join(sys.path[0], "data",  params["train_bug_code_index"]))

    print('data set size: ' + str(len(data_label)))
    print('num_pos: ' + str(num_pos))
    print("Pos/Neg: {:d}/{:d}".format(num_pos, len(data_label) - num_pos))
    print("Pos Ratio: {:f}".format(float(num_pos) / float(len(data_label))))

    # Build vocabulary
    print("Building vocabulary...")
    left_max_document_length = max([len(x.split(" ")) for x in data_left])
    left_vocab_processor = learn.preprocessing.VocabularyProcessor(left_max_document_length)
    x_left = np.array(list(left_vocab_processor.fit_transform(data_left)))

    right_max_document_length = max([len(str(y).split(" ")) for y in data_right])
    right_vocab_processor = learn.preprocessing.VocabularyProcessor(right_max_document_length)
    x_right = np.array(list(right_vocab_processor.fit_transform(data_right)))

    # Randomly shuffle data
    np.random.seed(params["seed"])
    shuffle_indices = np.random.permutation(np.arange(len(data_label)))
    data_left_shuffle = x_left[shuffle_indices]
    data_right_shuffle = x_right[shuffle_indices]
    data_label_shuffle = data_label[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    split_num = int(len(data_label) * params["eval_split"])
    x_left_train, x_left_dev = data_left_shuffle[:-split_num], data_left_shuffle[-split_num:]
    x_right_train, x_right_dev = data_right_shuffle[:-split_num], data_right_shuffle[-split_num:]
    y_train, y_dev = data_label_shuffle[:-split_num], data_label_shuffle[-split_num:]

    print("Left vocabulary Size: {:d}".format(len(left_vocab_processor.vocabulary_)))
    print("Right vocabulary Size: {:d}".format(len(right_vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    del data_left, data_right, x_left, x_right, data_left_shuffle, data_right_shuffle, data_label_shuffle

    with tf.Graph().as_default():
        # Imbalance class loss function weights
        ratio = float(num_pos) / float(len(data_label))
        class_weight = tf.constant(1 - ratio)
        # class_weight = ratio

        session_conf = tf.ConfigProto(
        allow_soft_placement=params["allow_soft_placement"],
        log_device_placement=params["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = BugCodeCNN(
                max_len_left=left_max_document_length,
                max_len_right=right_max_document_length,
                left_vocab_size=len(left_vocab_processor.vocabulary_),
                right_vocab_size=len(right_vocab_processor.vocabulary_),
                embedding_size=params["embedding_dim"],
                filter_sizes=list(map(int, params["filter_sizes"].split(","))),
                num_filters=params["num_filters"],
                num_hidden=params["num_hidden"],
                class_weights=class_weight,
                l2_reg_lambda=params["l2_reg_lambda"])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

            # Write vocabulary
            left_vocab_processor.save(os.path.join(out_dir, "left_vocab"))
            right_vocab_processor.save(os.path.join(out_dir, "right_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_left_batch, x_right_batch, y_batch):
                feed_dict = {
                    cnn.input_left: x_left_batch,
                    cnn.input_right: x_right_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params["dropout_keep_prob"]
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_left_batch_dev, x_right_batch_dev, y_batch_dev, writer=None):
                feed_dict = {
                    cnn.input_left: x_left_batch_dev,
                    cnn.input_right: x_right_batch_dev,
                    cnn.input_y: y_batch_dev,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, sims, pres, predictions = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.sims, cnn.scores, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

                # simple accuracy
                temp_predictions = predictions
                temp_data_label = y_batch_dev
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
                print("Total batch examples: {:d}, pos: {:d}, neg: {:d}".format(len(temp_data_label), temp_num_pos, temp_num_neg))

                print("Correct Accuracy: {:d}, {:d}, {:g}".format(correct_predictions, temp_num_pos, temp_num_pos_percent))
                print("Wrong Accuracy: {:d}, {:d}, {:g}".format(wrong_predictions, temp_num_neg, temp_num_neg_percent))
                print("Mean Accuracy: {:g}".format(temp_mean_percent))
                return loss, accuracy, temp_mean_percent

            def overfit(dev_loss):
                n = len(dev_loss)
                if n < 5:
                    return False
                for i in range(n-4, n):
                    if dev_loss[i] > dev_loss[i-1]:
                        return False
                return True

            # Generate batches
            batches = data_help.batch_iter(
                list(zip(x_left_train, x_right_train, y_train)), params["batch_size"], params["num_epochs"])

            # Training loop. For each batch...
            dev_loss = []
            mean_loss = []
            for batch in batches:
                x1_batch, x2_batch, y_batch = zip(*batch)
                train_step(x1_batch, x2_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params["evaluate_every"] == 0:
                    print("\nEvaluation:")
                    loss, accuracy, temp_mean_percent = dev_step(x_left_dev, x_right_dev, y_dev, writer=dev_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    dev_loss.append(accuracy)
                    mean_loss.append(temp_mean_percent)
                    print("\nRecently accuracy:")
                    print(dev_loss[-10:])
                    print(mean_loss[-10:])
                    # if overfit(mean_loss):
                    #     print('Overfit!!')
                    #     break
                    print("")
                if current_step % params["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    print(sys.path[0])
    main(os.path.join(sys.path[0], "./simple_training_config.json"))