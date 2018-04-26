#! /usr/bin/env python
# encoding=utf-8

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size],  -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + max-pool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
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
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # self.accuracy, _ = tf.metrics.mean_per_class_accuracy(labels=tf.argmax(
            #     self.input_y, 1), predictions=self.predictions, num_classes=2)

            # print(self.accuracy)


class BugCNN(object):
    """
    A CNN process the bug report natural language.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, vocab_size,
      embedding_size, filter_sizes, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + max-pool layer for each filter size
        self.pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                self.pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


class CodeCNN(object):
    """
    A CNN process the bug report natural language.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, vocab_size, embedding_size, statement_length, statement_filter_sizes, statement_num_filters,
            statement_num, code_filter_sizes, code_num_filters,):

        # Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.int32, [None, statement_num, statement_length], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.filter_sizes = code_filter_sizes
        self.num_filters = code_num_filters

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_y)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + max-pool layer for each filter size

        self.pooled_outputs = []
"""
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                self.pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
"""


class BugCodeCNN(object):
    def __init__(
      self, max_len_left, max_len_right, left_vocab_size, right_vocab_size,
      embedding_size, filter_sizes, num_filters, num_hidden, class_weights, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("left_embedding"):
            W = tf.Variable(
                tf.random_uniform([left_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_left), -1)
            print(self.embedded_chars_left)

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("right_embedding"):
            W = tf.Variable(
                tf.random_uniform([right_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_right), -1)
            print(self.embedded_chars_right)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_left,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_right,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_right.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(pooled_outputs_left, 3), [-1, num_filters_total], name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(pooled_outputs_right, 3), [-1, num_filters_total], name='h_pool_right')
        print(self.h_pool_left)
        print(self.h_pool_right)

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.h_pool_left, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_right), 1, keep_dims=True)
            print(self.sims)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Make input for classification
        self.new_input = tf.concat([self.h_pool_left, self.sims, self.h_pool_right], 1, name='new_input')

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2*num_filters_total+1, num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            print(self.h_drop)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # Imbalance class loss function weights
            # weighted_logits = tf.multiply(self.scores, class_weights)
            losses = tf.nn.weighted_cross_entropy_with_logits(logits=self.scores,
                                                              targets=self.input_y, pos_weight=class_weights)
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=weighted_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class BugCodeSplitCNN(object):
    def __init__(
      self, max_len_left, max_len_right, left_vocab_size, right_vocab_size,
      embedding_size, filter_sizes, num_filters, num_hidden, class_weights, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("left_embedding"):
            W = tf.Variable(
                tf.random_uniform([left_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_left), -1)
            print(self.embedded_chars_left)

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("right_embedding"):
            W = tf.Variable(
                tf.random_uniform([right_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_right), -1)
            print(self.embedded_chars_right)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_left,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_right,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len_right - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_right.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(pooled_outputs_left, 3), [-1, num_filters_total], name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(pooled_outputs_right, 3), [-1, num_filters_total], name='h_pool_right')
        print(self.h_pool_left)
        print(self.h_pool_right)

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.h_pool_left, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_right), 1, keepdims=True)
            print(self.sims)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Make input for classification
        self.new_input = tf.concat([self.h_pool_left, self.sims, self.h_pool_right], 1, name='new_input')

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2*num_filters_total+1, num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            print(self.h_drop)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # Imbalance class loss function weights
            # weighted_logits = tf.multiply(self.scores, class_weights)
            losses = tf.nn.weighted_cross_entropy_with_logits(logits=self.scores, targets=self.input_y, pos_weight=class_weights)
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=weighted_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
