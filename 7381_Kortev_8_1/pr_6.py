import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import layers, callbacks
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


import datetime

from var7 import gen_data


class Save_3_bests(callbacks.Callback):
    def __init__(self,val, prefix='model', metr='val_loss', date=datetime.datetime.now()):
        self.val=val
        self.prefix='{}_{}_{}_'.format(date.day,date.month,date.year)+prefix+'_'
        self.losses={}
        self.metr=metr

    def on_train_begin(self, logs=None):
        loss=self.model.evaluate(self.val[0],self.val[1])[0]
        self.losses[self.prefix+'1'] = loss
        self.losses[self.prefix+'2'] = loss
        self.losses[self.prefix+'3'] = loss
        for key in self.losses.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs={}):
        for (path,loss) in self.losses.items():
            if logs.get(self.metr) < loss:
                self.losses[path]=logs.get(self.metr)
                self.model.save(path)
                break

    def on_train_end(self, logs=None):
        for key in self.losses.keys():
            print(load_model(key).evaluate(self.validation_data[0],self.validation_data[1],verbose=0)[0])
            #0.652881709072325
            #0.6720317533281115
            #0.672176884677675


np.random.seed(999)

size=1500
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

num_ep=20

def build_model():
    model = Sequential()
    model.add(layers.Conv2D(22, (5, 5),strides=2 ,activation='relu', input_shape=(50, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(22, (3, 3), strides=2,activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model=build_model()

h=model.fit(x[:test],y[:test],epochs=num_ep,batch_size=1,verbose=2,validation_data=(x[test:],y[test:]), callbacks=[Save_3_bests((x[test:],y[test:]))])



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