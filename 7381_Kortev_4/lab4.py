import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers


(train_x, train_y),(test_x, test_y)=mnist.load_data()

train_x=train_x/255.0
test_x=test_x/255.0

train_y=to_categorical(train_y)
test_y=to_categorical(test_y)

model=Sequential()
model.add(Flatten(input_shape=(28,28,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

num_ep=50

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
h=model.fit(train_x,train_y,epochs=num_ep,batch_size=128,validation_split=0.15,verbose=0)

model.save('model.h5')

plt.plot(range(num_ep),h.history['val_accuracy'],'b',label='Val acc')
plt.plot(range(num_ep),h.history['accuracy'],'b--',label='train_acc')
plt.show()

plt.plot(range(num_ep),h.history['val_loss'],'b',label='Val loss')
plt.plot(range(num_ep),h.history['loss'],'b--',label='train_loss')
plt.show()
