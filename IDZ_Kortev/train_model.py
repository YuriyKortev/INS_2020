

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH '] = 'true'
os.system('del /S /Q /F log_dir\*') #clear logs
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from tensorflow.keras import callbacks
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from preprocess.preprocess import get_dataset

wrd2idx_path='preprocess/wrd2idx.txt'
pad=20  #len of sent
len_dict=30000  #len of dictionary
size_of_dataset=1000000
train=int(size_of_dataset*0.9)

(train_x, train_y), wrd2idx=get_dataset('preprocess/train_data.csv', wrd2idx_path, len_of_dir=len_dict-1, cut=pad)
train_y=train_y.astype('float32')


with open("log_dir/metadata.tsv", 'w',encoding='utf-8') as f: #creating metadata for tensorboard
    f.write("Index\tLabel\n")
    for key, ind in wrd2idx.items():
        f.write("{}\t{}\n".format(key,ind))

tb=callbacks.TensorBoard(
    embeddings_freq=1,
    log_dir='log_dir',
    histogram_freq=3
)


def train_dense_model():
    model = Sequential()
    model.add(Embedding(len_dict, 8, input_length=pad,name='emb'))
    model.add(Flatten())
    model.add(Dense(160, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(120, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 3;
    h = model.fit(train_x[:train], train_y[:train], epochs=num_epochs, batch_size=1024
                  , validation_data=(train_x[train:size_of_dataset], train_y[train:size_of_dataset]), verbose=2,
                  callbacks=[
                      callbacks.ReduceLROnPlateau(
                          monitor='val_loss',
                          factor=0.3,
                          patience=3
                      )
                  ])

    model.save('model.h5')

    plot_history(h,num_epochs)


def train_cnn_lstm():
    model = Sequential()
    model.add(Embedding(len_dict, 8, input_length=pad, name='emb'))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=48,kernel_size=5,strides=1,padding='valid',activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(130, dropout=0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 10;
    h = model.fit(train_x[:train], train_y[:train], epochs=num_epochs, batch_size=512
                  , validation_data=(train_x[train:size_of_dataset], train_y[train:size_of_dataset]), verbose=2,
                  callbacks=[
                      callbacks.ReduceLROnPlateau(
                          monitor='val_loss',
                          factor=0.3,
                          patience=6
                      )
                  ])

    plot_history(h,num_epochs)
    model.save('model_cnn_lstm.h5')

def train_2lstm():
    model = Sequential()
    model.add(Embedding(len_dict, 18, input_length=pad, name='emb'))
    model.add(LSTM(80, dropout=0.3,return_sequences=True))
    model.add(LSTM(150, dropout=0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 30;
    h = model.fit(train_x[:train], train_y[:train], epochs=num_epochs, batch_size=1024
                  , validation_data=(train_x[train:size_of_dataset], train_y[train:size_of_dataset]), verbose=2,
                  callbacks=[
                      callbacks.ReduceLROnPlateau(
                          monitor='val_loss',
                          factor=0.3,
                          patience=6
                      )
                  ])

    plot_history(h,num_epochs)
    model.save('model_2lstm.h5')

def train_bidir_rnn():
    model = Sequential()
    model.add(Embedding(len_dict, 18, input_length=pad, name='emb'))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(90, dropout=0.5)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    num_epochs = 12;
    h = model.fit(train_x[:train], train_y[:train], epochs=num_epochs, batch_size=1024
                  , validation_data=(train_x[train:size_of_dataset], train_y[train:size_of_dataset]), verbose=2,
                  callbacks=[
                      callbacks.ReduceLROnPlateau(
                          monitor='val_loss',
                          factor=0.2,
                          patience=6
                      )])

    plot_history(h, num_epochs)
    model.save('model_bidir_lstm.h5')

def plot_history(h, epochs):
    plt.plot(range(epochs), h.history['val_accuracy'], 'b-', label='val')
    plt.plot(range(epochs), h.history['accuracy'], 'b--', label='test')
    plt.title('test and val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()

    plt.plot(range(epochs), h.history['val_loss'], 'b-', label='val')
    plt.plot(range(epochs), h.history['loss'], 'b--', label='test')
    plt.title('test and val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


if __name__=='__main__':
    train_cnn_lstm()
