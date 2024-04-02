# from tensorflow_docs.vis import embed
from tensorflow import keras
from keras.utils import plot_model
from imutils import paths

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import csv
from scipy import ndimage
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from flirvideo import FlirVideo
import gc
R = 136
C = 144
SEQ_LENGTH = 100
NUM_FEATURES = 2048
EPOCHS = 1500
dataset_folder = "./A-W-Wave/*/*.ats"
video_paths = sorted(glob.glob(f"{dataset_folder}", recursive=True))





#%%

def load_video(Temp, max_frames=0, resize=(R, C)):
    min = Temp.min()
    max = Temp.max()

    Temp = 255*(Temp-min)/(max-min)

    frames = []
    len = Temp.shape[2]
    
    for i in range(len):
        size = Temp.shape

        if size[0] < resize[0]:
            d1 = np.int16(np.floor((resize[0]-size[0])/2))
            d2 = np.int16(resize[0]-size[0]-d1)

            T = np.append(np.zeros((d1, size[1])), np.append(Temp[:,:, i], np.zeros((d2, size[1])),axis=0),axis=0)
        else:
            d1 = np.int16(np.floor((size[0]-resize[0])/2))
            T = Temp[d1:d1+resize[0], :, i]

        if size[1] < resize[1]:
            d1 = np.int16(np.floor((resize[1]-size[1])/2))
            d2 = np.int16(resize[1]-size[1]-d1)

            T = np.append(np.zeros((resize[0],d1)), np.append(T, np.zeros((resize[0],d2)),axis=1),axis=1)
        else:
            d1 = np.int16(np.floor((size[1]-resize[1])/2))
            T = T[:, d1:d1+resize[1]]


        frame = np.dstack((T[:,:], np.zeros(resize), np.zeros(resize)))

        frames.append(frame)

    return np.array(frames)
#%%
def generateRandomData(Temp):
    angle = 360*np.random.random()
    #Temp = self.Temp
    Temp = ndimage.rotate(Temp, angle=angle, reshape=False, mode='nearest')
    if np.random.random()>=0.5:
        Temp = np.fliplr(Temp)
    if np.random.random()>=0.5:
        Temp = np.flipud(Temp)
    Temp = Temp+1*(np.random.random(Temp.shape)-0.5)

    return Temp

#%%
lables = []
data = []
for video in video_paths:
    print(video)
    v =FlirVideo(video)
    v.videoCutbatch(SEQ_LENGTH)
    s = v.Tempbatshe
    for a in s:
        a = load_video(a)
        data.append(a)
    lable = video.split('/')[-2]
    #lables.append()
    lables.extend([lable]*len(s))
    del v.Temp
    del v.Tempbatshe
    del v
    del s
    del a
    gc.collect()


np.save('data.npy',data)   
np.save('lables.npy', lables)
    
#restart
#%%
data= np.load('data.npy')   
lables = np.load('lables.npy')


X_train, X_test, y_train, y_test = train_test_split(data, lables, test_size=0.33, random_state=42)

np.save('X_train.npy',X_train)   
np.save('X_test.npy', X_test)
np.save('y_train.npy',y_train)   
np.save('y_test.npy', y_test)

#restart

#%%
X_train = np.load('X_train.npy')   
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')   
y_test  = np.load('y_test.npy')
#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train_number = le.transform(y_train)
y_test_number = le.transform(y_test)
#%%
b = len(X_train)
for i in range(b):
    c = generateRandomData(X_train[i])
    X_train = np.append(X_train,[c],0)
    y_train = np.append(y_train,y_train[i])
    print(i)


#%%
MAX_SEQ_LENGTH = SEQ_LENGTH
NUM_FEATURES = 2048
              
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(R, C, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((R, C, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()
#%%
i=0
num_samples = len(X_train)
MAX_SEQ_LENGTH = SEQ_LENGTH
frame_masks_train = np.ones(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
frame_features_train = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
for x in X_train:
    frame_features_train[i] = feature_extractor(x)
    i = i+1

#%%
i=0
num_samples = len(X_test)
MAX_SEQ_LENGTH = SEQ_LENGTH
frame_masks_test = np.ones(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
frame_features_test = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
for x in X_test:
    frame_features_test[i] = feature_extractor(x)
    i = i+1

np.save('frame_features_train.npy',frame_features_train)   
np.save('frame_features_test.npy', frame_features_test)
#%%
num_samples = len(X_train)
MAX_SEQ_LENGTH = SEQ_LENGTH
frame_masks_train = np.ones(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
frame_features_train = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

num_samples = len(X_test)
MAX_SEQ_LENGTH = SEQ_LENGTH
frame_masks_test = np.ones(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
frame_features_test = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

frame_features_train = np.load('frame_features_train.npy')   
frame_features_test  = np.load('frame_features_test.npy')
#%%
print(f"Frame features in train set: {frame_features_train.shape[0]}")
print(f"Frame masks in train set: {frame_features_test.shape[0]}")

MAX_SEQ_LENGTH = SEQ_LENGTH


def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    # x = keras.layers.GRU(16, return_sequences=True)(
    #     frame_features_input, mask=mask_input
    # )
    # x = keras.layers.GRU(8)(x)
    # x = keras.layers.Dropout(0.6)(x)
    # x = keras.layers.Dense(8, activation="relu")(x)
    # output = keras.layers.Dense(1, activation="relu")(x)


    # Questo mi piace
    # x = keras.layers.GRU(24, return_sequences=True)(
    #     frame_features_input, mask=mask_input
    # )
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.GRU(12)(x)
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(8, activation="relu")(x)
    # x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.GRU(24, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.GRU(18)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(3, activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy"
    )
    return rnn_model

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [diam]')
    plt.legend()
    plt.grid(True)
    plt.savefig('history.pdf')

def run_experiment():
    filepath = "./A-W-Wave"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    print(seq_model.summary())
    # plot_model(seq_model, to_file='model.png', show_shapes=True, show_layer_names=True)
    history = seq_model.fit(
        [frame_features_train, frame_masks_train],
        y_train_number,
        validation_split=0.2,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    plot_loss(history)
    seq_model.save('model.h5')

    # history = []

    seq_model.load_weights(filepath)
   # _, mse = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
   # print(f"Validation MAE {mse}")

    return history, seq_model


_, sequence_model = run_experiment()
filepath = "./A-W-Wave"
seq_model = get_sequence_model()
seq_model.load_weights(filepath)
loss, accuracy = seq_model.evaluate([frame_features_test, frame_masks_test], y_test_number)

















