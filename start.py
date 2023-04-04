# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np 
import pandas as pd 
import os
# for dirname, _, filenames in os.walk('D:\\Program Files (x86)\\program\\UNSW-NB15 - CSV Files'):
    # for filename in filenames:
    #     print(os.path.join(dirname, filename))
 

import sys
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#set path
df = pd.read_csv('D:\\Program Files (x86)\\program\\UNSW-NB15 - CSV Files\\UNSW-NB15 - CSV Files\\a part of training and testing set\\UNSW_NB15_testing-set.csv')
# print(df.head())
qp = pd.read_csv('D:\\Program Files (x86)\\program\\UNSW-NB15 - CSV Files\\UNSW-NB15 - CSV Files\\a part of training and testing set\\UNSW_NB15_training-set.csv')
# print(qp)


#Dropping the last columns of training set
# df = df.drop('id', 1)
df = df.drop('id', axis=1)
 # we don't need it in this project
# print(df.shape)
df = df.drop('label', axis=1) # we don't need it in this project
# print(df.head())


#Dropping the last columns of testing set
qp = qp.drop('id', axis=1)
qp = qp.drop('label', axis=1)

# dict(x_test_2)
#     pred = np.argmax(pred,axis=1)
#     y_eval = np.argmax(y_test_2,axis=1)
#     score = metrics.accuracy_score(y_eval, pred)
#     oos_pred.append(score)
#     print("Validation score: {}".format(score))


#defining col list
cols = ['proto','state','service']

#One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, axis=1)
    return df


#Merging train and test data
combined_data = pd.concat([df,qp])
# print(combined_data)
tmp = combined_data.pop('attack_cat')
#Applying one hot encoding to combined data
combined_data = one_hot(combined_data,cols)


#Function to min-max normalize  归一化处理
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


#Normalizing training set
new_train_df = normalize(combined_data,combined_data.columns)
#print(new_train_df)
new_train_df["Class"] = tmp
y_train=new_train_df["Class"]
# print(y_train)
combined_data_X = new_train_df.drop('Class', axis=1)
# print(combined_data_X)



#模型定义
oos_pred = []
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=42)
# StratifiedKFold()的用法：分层采样
# print(kfold.get_n_splits(combined_data_X,y_train))
batch_size = 32

model = Sequential()
#定义卷积层
model.add(Convolution1D(64, kernel_size=64, padding="same",activation="relu",input_shape=(196, 1)))
#定义池化层
model.add(MaxPooling1D(pool_size=(10)))
#定义归一化层
model.add(BatchNormalization())
#双向LSTM
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Reshape((128, 1), input_shape = (128, )))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
#model.add(Reshape((128, 1), input_shape = (128, )))
#丢弃法，防止过拟合
model.add(Dropout(0.6))
#全连接层
model.add(Dense(10))
#softmax 回归，在全连接层上
model.add(Activation('softmax'))
#定义loss函数和优化函数adam
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# categorical_crossentropy 是一种用于多分类问题的损失函数。它可以用来衡量预测概率分布与真实概率分布之间的差距。
# 这种损失函数使用了对数损失，其中预测概率分布的对数与真实概率分布的对数之差是最小的。



# /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:3:
#  UserWarning: Update your `Conv1D` call to the Keras 2 API: 
# `Conv1D(64, kernel_size=64, activation="relu", input_shape=(196, 1), padding="same")`
#   This is separate from the ipykernel package so we can avoid doing imports until
# /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:4:
#  UserWarning: Update your `MaxPooling1D` call to the Keras 2 API: `MaxPooling1D(pool_size=10)`
#   after removing the cwd from sys.path.
# /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:8:
#  UserWarning: Update your `MaxPooling1D` call to the Keras 2 API: `MaxPooling1D(pool_size=5)`
# for layer in model.layers:
#     print(layer.output_shape)
# print(model.summary())


# pred
# array([3, 6, 6, ..., 6, 6, 6])
# oos_pred
# [0.8054130412847241, 0.811473501195318]
# test_y.value_counts()
# Normal            46500
# Generic           29435
# Exploits          22262
# Fuzzers           12123
# DoS                8177
# Reconnaissance     6994
# Analysis           1339
# Backdoor           1164
# Shellcode           755
# Worms                87
# Name: Class, dtype: int64

#模型训练
for train_index, test_index in kfold.split(combined_data_X,y_train):
    train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
    

    print("train index:",train_index)
    print("test index:",test_index)
    print(train_y.value_counts())
    

    train_X_over,train_y_over= oversample.fit_resample(train_X, train_y)
    print(train_y_over.value_counts())
    

    x_columns_train = new_train_df.columns.drop('Class')
    x_train_array = train_X_over[x_columns_train].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
    

    dummies = pd.get_dummies(train_y_over) # Classification
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values
    
    
    # x_columns_train = new_train_df.columns.drop('Class')
    # x_train_array = train_X_over[x_columns_train].values
    # x_train_1=np.res

    x_columns_test = new_train_df.columns.drop('Class')
    x_test_array = test_X[x_columns_test].values
    x_test_2=np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))
    

    dummies_test = pd.get_dummies(test_y) 
    # Classification
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values
    
   
    model.fit(x_train_1, y_train_1,validation_data=(x_test_2,y_test_2), epochs=15)
    #实现测试
    pred = model.predict(x_test_2)
    pred = np.argmax(pred,axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    #模型accuracy
    score = metrics.accuracy_score(y_eval, pred)
    oos_pred.append(score)
    print("Validation score: {}".format(score))


#     dummies_test.columns
#     Index(['Analysis', 'Backdoor', 'DoS', 
#      'Exploits', 'Fuzzers', 'Generic',
#        'Normal', 'Reconnaissance', 'Shellcode', 'Worms'],
#       dtype='object')

