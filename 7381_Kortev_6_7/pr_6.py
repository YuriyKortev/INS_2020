import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import layers
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from var7 import gen_data

np.random.seed(999)

size=1000
x,y=gen_data(size=size)
ran=list(range(len(x)))
np.random.shuffle(ran)
x=x[ran]
y=y[ran]
x=x.reshape((size,50,50,1))


encoder=LabelEncoder()
encoder.fit(y)
y=encoder.transform(y)
y=to_categorical(y)

test=int(0.7*size)

num_ep=4

model=Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(50,50,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
h=model.fit(x[:test],y[:test],epochs=num_ep,batch_size=1,verbose=1,validation_data=(x[test:],y[test:]))



plt.plot(range(num_ep),h.history['val_accuracy'],label='val')
plt.plot(range(num_ep),h.history['accuracy'],'b--',label='test')
plt.title('test anv val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

plt.plot(range(num_ep),h.history['val_loss'],label='val')
plt.plot(range(num_ep),h.history['loss'],'b--',label='test')
plt.title('test anv val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()



