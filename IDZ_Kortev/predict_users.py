import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import numpy as np
from sys import argv
from preprocess.preprocess import indexing,gen_tokens
from tensorflow.keras.preprocessing import sequence
import json

class PredictSentiment(object):   #class essembling models
    def __init__(self):
        self.cnn_lstm = load_model('model_cnn_lstm.h5')
        self.model_2lstm = load_model('model_2lstm.h5')
        self.fnn = load_model('model.h5')
        self.bi_cnn = load_model('model_bidir_lstm.h5')

    def predict(self,x):
        cnn_lstm_pred = self.cnn_lstm.predict(x)
        fnn_pred = self.fnn.predict(x)
        bi_cnn_pred = self.bi_cnn.predict(x)
        lstm_pred = self.model_2lstm.predict(x)

        pred=0.3*lstm_pred+0.24*bi_cnn_pred+0.23*fnn_pred+0.23*cnn_lstm_pred

        for i, pr in enumerate(pred):
            if 0.0<=pr[0]<0.4:
                print('{}: negativ'.format(i + 1), end=' ')
            elif 0.4<=pr[0]<0.6:
                print('{}: neitral'.format(i + 1), end=' ')
            elif 0.6 <= pr[0] <= 1.0:
                print('{}: positiv'.format(i + 1), end=' ')

            print('({:.2f}=fnn: {:.2f} + cnn_lstm: {:.2f} + bi_lstm: {:.2f} + 2lstm: {:.2f})'.format(pr[0], fnn_pred[i][0],
                                                                                         cnn_lstm_pred[i][0],
                                                                                         bi_cnn_pred[i][0],
                                                                                                     lstm_pred[i][0]))
ensemble_predicter=PredictSentiment()
twits= argv[1:]

with open("preprocess/wrd2idx.txt", 'r') as json_file:
    wrd2idx = json.load(json_file)

wrd2idx = dict(list(wrd2idx.items())[:30000])

for i, nof in enumerate(twits):
    with open(nof,'r') as file:
        twits[i]=file.read()

twits=gen_tokens(twits)
print('tokens: \n', twits)

for i, sent in enumerate(twits):  #indexing tokens
    for j, word in enumerate(sent):
        if word in wrd2idx:
            twits[i][j]=wrd2idx[word]
        else:
            twits[i][j]=0

print('word indexing: \n', twits)
twits=sequence.pad_sequences(twits, maxlen=20)

ensemble_predicter.predict(twits)

