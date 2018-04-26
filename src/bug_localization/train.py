#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_help
from merge_cnn_cnn import NPCNN
from tensorflow.contrib import learn
import json


def train_cnn(training_config_file=None):

    params = json.loads(open(training_config_file).read())

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y_text, label = data_help.load_text_and_label(params["bug_text_file"], params["code_text_file"], params["bug_code_index_file"])

    # Build vocabulary
    x_max_document_length = max([len(x.split(" ")) for x in x_text])
    x_vocab_processor = learn.preprocessing.VocabularyProcessor(x_max_document_length)
    x = np.array(list(x_vocab_processor.fit_transform(x_text)))

    y_max_document_length = max([len(str(y).split(" ")) for y in y_text])
    y_vocab_processor = learn.preprocessing.VocabularyProcessor(y_max_document_length)
    y = np.array(list(y_vocab_processor.fit_transform(y_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    label_shuffled = label[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(params["dev_sample_percentage"] * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    label_train, label_dev = label[:dev_sample_index], label_shuffled[dev_sample_index:]

    del x, y, label, x_shuffled, y_shuffled, label_shuffled

    print("Bug Vocabulary Size: {:d}".format(len(x_vocab_processor.vocabulary_)))
    print("Code Vocabulary Size: {:d}".format(len(y_vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=params["allow_soft_placement"],
          log_device_placement=params["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            merge_cnn = NPCNN(
                x_sequence_length=x_train.shape[1],
                x_vocab_size=len(x_vocab_processor.vocabulary_),
                x_filter_sizes=list(map(int, params["filter_sizes"].split(","))),
                x_num_filters=params["num_filters"],
                x_embedding_size=params["embedding_dim"],
                y_sequence_length=x_train.shape[1],
                y_vocab_size=len(y_vocab_processor.vocabulary_),
                y_filter_sizes=list(map(int, params["filter_sizes"].split(","))),
                y_num_filters=params["num_filters"],
                y_embedding_size=params["embedding_dim"],
                num_classes=label_train.shape[1],
                l2_reg_lambda=params["l2_reg_lambda"])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(merge_cnn.loss)
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
            loss_summary = tf.summary.scalar("loss", merge_cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", merge_cnn.accuracy)

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
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=params["num_checkpoints"])

            # Write vocabulary
            x_vocab_processor.save(os.path.join(out_dir, "bug_vocab"))
            y_vocab_processor.save(os.path.join(out_dir, "code_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, label_batch):
                """
                A single training step
                """
                # print(x_batch, y_batch, label_train)
                feed_dict = {
                    merge_cnn.bug_cnn.input_x: x_batch,
                    merge_cnn.code_cnn.input_x: y_batch,
                    merge_cnn.bug_cnn.dropout_keep_prob: params["dropout_keep_prob"],
                    merge_cnn.code_cnn.dropout_keep_prob: params["dropout_keep_prob"],
                    merge_cnn.dropout_keep_prob: params["dropout_keep_prob"],
                    merge_cnn.input_label: label_batch,
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, merge_cnn.loss, merge_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, label_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                # print(x_batch, y_batch, label_train)
                feed_dict = {
                    merge_cnn.bug_cnn.input_x: x_batch,
                    merge_cnn.code_cnn.input_x: y_batch,
                    merge_cnn.bug_cnn.dropout_keep_prob: 1.0,
                    merge_cnn.code_cnn.dropout_keep_prob: 1.0,
                    merge_cnn.dropout_keep_prob: 1.0,
                    merge_cnn.input_label: label_batch,
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, merge_cnn.loss, merge_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_help.batch_iter(
                list(zip(x_train, y_train, label_train)), params["batch_size"], params["num_epochs"])
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch, label_batch = zip(*batch)
                train_step(x_batch, y_batch, label_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % params["evaluate_every"] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, label_dev, writer=dev_summary_writer)
                    print("")
                if current_step % params["checkpoint_every"] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    train_cnn("./training_config")