import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from src.nn.nn_model_builder import build
from src.nn.nn_model_trainer import compile_fit, saveModels

train_data_precentage = .9
MINI_BATCH_SIZE = 16

input_data_paths = {
    'stop' : '/Users/hbojja/uiuc/CS445-CP/FinalProject/train_data_individual/cleansed/stop',
    'school_zone' : '/Users/hbojja/uiuc/CS445-CP/FinalProject/train_data_individual/cleansed/school_zone',
    'speed_limit_25' : '/Users/hbojja/uiuc/CS445-CP/FinalProject/train_data_individual/cleansed/speed_limit_25',
    'speed_limit_35' : '/Users/hbojja/uiuc/CS445-CP/FinalProject/train_data_individual/cleansed/speed_limit_35',
    'other' : '/Users/hbojja/uiuc/CS445-CP/FinalProject/train_data_individual/cleansed/other'
}

X = []
Y = []
labels = []
for sign in input_data_paths:
    dir_path = input_data_paths[sign]
    for filename in os.listdir(dir_path):
        if not filename.endswith('.jpg'):
            continue

        # 0 - grey scale, 1 - BGR
        image = cv2.imread(dir_path + '/' + filename, 0)

        # X.append(image)
        X.append(image.ravel())
        labels.append(sign)
        # print(image.shape)

lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(labels)

X = np.array(X)

print(X.shape)
scaler = MinMaxScaler()
X = scaler.fit_transform(X, y=None)

train_set_X, val_set_X, train_set_Y, val_set_Y = train_test_split(X, Y, test_size=1-train_data_precentage, shuffle=True)

train_set_X, val_set_X, train_set_Y, val_set_Y = np.array(train_set_X), np.array(val_set_X), np.array(train_set_Y), np.array(val_set_Y )
train_set = tf.data.Dataset.from_tensor_slices((train_set_X, train_set_Y))
train_set = train_set.shuffle(1000).batch(MINI_BATCH_SIZE).repeat()

val_set = tf.data.Dataset.from_tensor_slices((val_set_X, val_set_Y))
val_set = val_set.batch(MINI_BATCH_SIZE).repeat()

neural_network = build(X[0].shape[0], len(input_data_paths), train_set_X[0].shape)

model, model_history = compile_fit(neural_network, train_set, val_set, train_set_X.shape[0], val_set_X.shape[0], MINI_BATCH_SIZE)


saveModels(model, scaler, lb, '/Users/hbojja/uiuc/CS445-CP/FinalProject/trained_models')