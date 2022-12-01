#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:46:46 2022

@author: danielalavi
"""

#%% PACKAGES

import numpy as np
import pandas as pd
import keras as k
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import InputLayer
from keras import models
from keras.utils import to_categorical
from keras.regularizers import l1

#%% GET DATA

df_test = pd.read_csv("test.csv", dtype = float)
df_train = pd.read_csv("train.csv", dtype = float)


#%% DATA PREPARATION

# separate train and test
X_train = df_train.iloc[:, 1:]
X_train = np.array(X_train)
y_train = df_train.iloc[:, 0]

# encode y as categorical using keras
y_train = to_categorical(y_train, 10)

#%% CREATE NEURAL NETWORK

# specify the model
model = models.Sequential([
    InputLayer(input_shape = X_train.shape[1]),
    # add layers to the model
    Dense(200, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(50, activation = 'relu'),
    Dense(10, activation = 'softmax')
    ])    

# compile model
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

# check the summary
print(model.summary())

# train the model
model.fit(X_train, y_train,
          epochs = 30)

score = model.evaluate(X_train, y_train, verbose = 0)

print('The accuracy of the model was: ', score[1])

#%% PREDICT TEST DATA

X_test = np.array(df_test)

y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis = 1)

#%% CREATE SUBMISSION FILE

df_sub = pd.Series(y_hat, name = 'Label')
df_id = pd.Series(range(1,28001), name = "ImageId")

submission = pd.concat([df_id, df_sub], axis = 1)

submission.to_csv("submission5.csv", index = False)
