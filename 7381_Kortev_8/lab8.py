import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import layers, callbacks
from keras.optimizers import RMSprop
from keras.models import Sequential, load_model

import sys

import random
random.seed(999)



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class Gen_text(callbacks.Callback):
    def __init__(self, text_seed):
        self.text_seed=text_seed

    def on_epoch_end(self, epoch, logs=None):

        if(epoch==20 or epoch==30 or epoch==60 or True):
            for temp in [0.5,0.01]:
                print("temp: ",temp)
                gen_text = self.text_seed[:]
                sys.stdout.write(gen_text)
                for i in range(400):
                    sampled = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(gen_text):
                        sampled[0, t, char_indices[char]] = 1

                    pred = self.model.predict(sampled, verbose=0)[0]
                    next_char = chars[sample(pred, temp)]

                    gen_text += next_char
                    gen_text = gen_text[1:]
                    sys.stdout.write(next_char)
                    if(i%100==0):
                        print()

                print('\n------\n')

text_seed="'Never!' said the Queen furiously".lower()

maxlen=len(text_seed)
step=3

text=open("wonderland.txt").read().lower()
chars=sorted(list(set(text)))
char_indices=dict((char,chars.index(char)) for char in chars)


text_seed_ind=random.randint(0,len(text)-maxlen-1)

senteces=[]
next_chars=[]


for i in range(0,len(text)-maxlen,step):
    senteces.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

x=np.zeros((len(senteces),maxlen,len(chars)),dtype=np.bool)
y=np.zeros((len(senteces),len(chars)), dtype=np.bool)

for i,sentece in enumerate(senteces):
    for t, char in enumerate(sentece):
        x[i,t,char_indices[char]]=1
    y[i, char_indices[next_chars[i]]]=1


filepath="weights-improvement.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

model = Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(x,y,batch_size=128,epochs=100,callbacks=[Gen_text(text_seed=text_seed),checkpoint],verbose=2)
