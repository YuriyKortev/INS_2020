import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from keras.datasets import imdb

num_wprds=10000
num_epochs=2

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=num_wprds)
data=np.concatenate((train_data,test_data),axis=0)
labels=np.concatenate((train_labels,test_labels),axis=0)

def vectorize_sequences(secuences, dimension=num_wprds):
    results=np.zeros((len(secuences),dimension))
    for i, secuence in enumerate(secuences):
        results[i, secuence]=1.
    return results


data=vectorize_sequences(data)
labels=np.asarray(labels).astype('float32')

model=Sequential()
model.add(Dense(16, activation = "relu", input_shape=(10000, )))
model.add(Dropout(0.3))
model.add(Dense(16, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

h=model.fit(data[10000:],labels[10000:], epochs=num_epochs,batch_size=512,validation_data=(data[:10000],labels[:10000]))


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



