import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

num_wprds=10000

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=num_wprds)
data=np.concatenate((train_data,test_data),axis=0)
labels=np.concatenate((train_labels,test_labels),axis=0)

data=sequence.pad_sequences(data,maxlen=500)
labels=np.asarray(labels).astype('float32')

cnn_lstm=load_model('model_cnn_lstm.h5')
fnn=load_model('model.h5')
bi_cnn=load_model('model_bi.h5')

lstm_pred=cnn_lstm.evaluate(data[5000:10000],labels[5000:10000])
fnn_pred=fnn.evaluate(data[5000:10000],labels[5000:10000])
bi_pred=bi_cnn.evaluate(data[5000:10000],labels[5000:10000])
print((lstm_pred[1]+fnn_pred[1]+bi_pred[1])/3)