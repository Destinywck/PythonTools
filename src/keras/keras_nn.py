#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
# from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from utils.tools import calculate_performace, draw_roc, draw_pr
import random
from sklearn.preprocessing import StandardScaler


def model_init():
    model = Sequential()
    model.add(Dense(768, input_shape=(148,), init='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, init='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, init='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='he_uniform'))

    model.compile(optimizer='adam', loss='mse')
    # optimizer: sgd, rmsprop ,adagrad,adadelta,adam,adamax,nadam
    # loss: mse,mae,mape, msle ,squared_hinge,hinge,binary_crossentropy,categorical_crossentropy
    #      sparse_categorical_crossentrop,kullback_leibler_divergence,poisson,cosine_proximity
    # model.summary()

    return model


def start_fit(dataSet):
    index = [i for i in range(len(dataSet))]
    random.shuffle(index)
    data = dataSet[index]
    X = dataSet[:, 0:148]
    Y = dataSet[:, 148]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # normalization
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dbn_model = KerasClassifier(model_init, epochs=500, batch_size=64, verbose=0)
    dbn_model.fit(X_train, y_train)
    y_ped = dbn_model.predict(X_test)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(len(y_ped), y_ped, y_test)
    print('DBN:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,roc_auc=%f'
          % (acc, precision, npv, sensitivity, specificity, mcc, roc_auc))


# 取数据
dataSet = np.loadtxt('D:\Program\example\RBPP11\DBN\RBP_9857.csv', delimiter=',')
start_fit(dataSet)






