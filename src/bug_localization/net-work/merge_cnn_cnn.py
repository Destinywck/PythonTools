#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from bug_code_cnn import BugCNN
from bug_code_cnn import CodeCNN


class NPCNN(object):

    def __init__(
      self, x_sequence_length, x_vocab_size, x_filter_sizes, x_num_filters, x_embedding_size,
            y_sequence_length, y_vocab_size, y_filter_sizes, y_num_filters, y_embedding_size,
            num_classes, l2_reg_lambda=0.0):
        # Placeholders for input_label, dropout_keep_prob
        self.input_label = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.bug_cnn = BugCNN(
            sequence_length=x_sequence_length,
            vocab_size=x_vocab_size,
            embedding_size=x_embedding_size,
            filter_sizes=x_filter_sizes,
            num_filters=x_num_filters)

        self.code_cnn = CodeCNN(
            sequence_length=y_sequence_length,
            vocab_size=y_vocab_size,
            embedding_size=y_embedding_size,
            filter_sizes=y_filter_sizes,
            num_filters=y_num_filters)

        # Combine all the pooled features
        num_filters_total = self.bug_cnn.num_filters * len(self.bug_cnn.filter_sizes) + self.code_cnn.num_filters * len(self.code_cnn.filter_sizes)
        pooled_outputs = []
        pooled_outputs.append(self.bug_cnn.pooled_outputs)
        pooled_outputs.append(self.code_cnn.pooled_outputs)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_label)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # self.accuracy, _ = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(
            #     self.input_y, 1), predictions=self.predictions, num_classes=2)

            # print(self.accuracy)