import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets.imdb import get_word_index
from sys import argv
import numpy as np

from keras.preprocessing.text import text_to_word_sequence

def vectorize_sequences(secuences, dimension=10000):
    results=np.zeros((len(secuences),dimension))
    for i, secuence in enumerate(secuences):
        results[i, secuence]=1.
    return results

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

reviews=vectorize_sequences(reviews)

model=load_model('model.h5')

prediction=model.predict_classes(reviews)

print("I'm very smart neuron network! And i think what...")
for i, pr in enumerate(prediction):
    if pr[0]==1:
        print('Film №{} is a Great movie!'.format(i+1))
    else:
        print('Film №{} is a horrible movie!!!'.format(i+1))

