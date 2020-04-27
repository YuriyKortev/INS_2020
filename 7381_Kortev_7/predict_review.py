import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets.imdb import get_word_index
from sys import argv
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

def filter(list_of_revs):
    for i in range(len(list_of_revs)):
        j=0
        while j<len(list_of_revs[i]):
            if list_of_revs[i][j] > 10000:
                del list_of_revs[i][j]
            else:
                j+=1

def decode(rev):
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in rev])
    return decoded_review


reviews=argv[1:]
word_index=get_word_index()

for i, nameOfFile in enumerate(reviews):
    with open(nameOfFile, 'r') as file:
        strings = text_to_word_sequence(file.read())
    for j, word in enumerate(strings):
        if word in word_index:
            #print(word, word_index[word])
            strings[j]=word_index[word]+3
        else:
            strings[j]=2
    reviews[i]=strings


filter(reviews)

reviews=sequence.pad_sequences(reviews,maxlen=500)

cnn_lstm=load_model('model_cnn_lstm.h5')
fnn=load_model('model.h5')
bi_cnn=load_model('model_bi.h5')

cnn_lstm_pred=cnn_lstm.predict(reviews)
fnn_pred=fnn.predict(reviews)
bi_cnn_pred=bi_cnn.predict(reviews)


prediction=(cnn_lstm_pred+fnn_pred+bi_cnn_pred)/3

print("I'm very smart neuron network! And i think what...")
for i, pr in enumerate(prediction):
    if np.round(pr[0])==1:
        print('Film №{} is a Great movie!'.format(i+1),'({:.2f}=fnn: {:.2f} + cnn_lstm: {:.2f} + bi_lstm: {:.2f})'.format(pr[0],fnn_pred[i][0],cnn_lstm_pred[i][0],bi_cnn_pred[i][0]))
    else:
        print('Film №{} is a horrible movie!!!'.format(i+1),'({:.2f}=fnn: {:.2f} + cnn_lstm: {:.2f} + bi_lstm: {:.2f})'.format(pr[0],fnn_pred[i][0],cnn_lstm_pred[i][0],bi_cnn_pred[i][0]))

