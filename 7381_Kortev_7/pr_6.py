import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D,GlobalMaxPool1D, Bidirectional
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.utils import to_categorical

from keras.preprocessing import sequence

from keras.datasets import imdb

num_wprds=10000
num_epochs=4

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=num_wprds)
data=np.concatenate((train_data,test_data),axis=0)
labels=np.concatenate((train_labels,test_labels),axis=0)

data=sequence.pad_sequences(data,maxlen=500, truncating='post')

labels=np.asarray(labels).astype('float32')

model=Sequential()
model.add(Embedding(num_wprds,3,input_length=500))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

h=model.fit(data[10000:],labels[10000:], epochs=num_epochs,batch_size=512
            ,validation_data=(data[:5000],labels[:5000]),verbose=2)


model.save('model.h5')

plt.plot(range(num_epochs),h.history['val_accuracy'],'b-',label='val')
plt.plot(range(num_epochs),h.history['accuracy'],'b--',label='test')
plt.title('test and val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

plt.plot(range(num_epochs),h.history['val_loss'],'b-',label='val')
plt.plot(range(num_epochs),h.history['loss'],'b--',label='test')
plt.title('test and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()