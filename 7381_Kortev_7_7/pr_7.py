import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import layers, callbacks
from keras.models import Sequential
from matplotlib import pyplot as plt

from var7 import gen_sequence

def gen_data_from_sequence(seq_len = 1006, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)

class Predict_on_ep_end(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predicted_res = self.model.predict(test_data)
        pred_length = range(len(predicted_res))
        if(True):
            plt.plot(pred_length, predicted_res, '--', label='predicted')
            plt.plot(pred_length, test_res, '-', label='res')
            plt.title('ep={}'.format(epoch))
            plt.show()

num_epochs=40
data, res = gen_data_from_sequence()


dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(150,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
model.add(layers.LSTM(150,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.2))
model.add(layers.GRU(150,input_shape=(None,1),recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse',metrics=['mae'])
h = model.fit(train_data,train_res,epochs=num_epochs,validation_data=(val_data, val_res),callbacks=[Predict_on_ep_end(),
                                                                                                    callbacks.ReduceLROnPlateau(
                                                                                                        monitor='val_loss',
                                                                                                        factor=0.1,
                                                                                                        patience=4
                                                                                                    ),
                                                                                                    callbacks.EarlyStopping(
                                                                                                        monitor='val_loss',
                                                                                                        patience=10
                                                                                                    )])


predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length,predicted_res,'b',label='predicted')
plt.plot(pred_length,test_res,'b--',label='res')
plt.show()


plt.plot(range(len(h.history['val_mae'])),h.history['val_mae'],'b-',label='val')
plt.plot(range(len(h.history['mae'])),h.history['mae'],'b--',label='test')
plt.title('test and val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

plt.plot(range(len(h.history['val_loss'])),h.history['val_loss'],'b-',label='val')
plt.plot(range(len(h.history['loss'])),h.history['loss'],'b--',label='test')
plt.title('test and val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()