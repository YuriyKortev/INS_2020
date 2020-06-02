import pandas as pd
import numpy as np
import  re
import nltk
import spacy
from nltk import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import words
import json
import random
from keras.preprocessing import sequence
import math

random.seed(999)

word2idx_path='wrd2idx.txt'

tag_dict = {"J": wordnet.ADJ,    #dictionary for lemmatizer
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp=re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl=r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word=self.repeat_regexp.sub(self.repl,word)
        if repl_word!=word:
            return self.replace(repl_word)
        else:
            return repl_word

def gen_tokens(X):
    stop_words = set(stopwords.words("english")) #stopwords like 'i we their'
    tweet = TweetTokenizer()
    lemmat = WordNetLemmatizer()
    repeat_replacer=RepeatReplacer()

    for i, x in enumerate(X):
        x = re.sub(r'https?://[^\"\s>]+', ' ', x) #del urls
        x = re.sub(r'@\w+', ' ', x)              #del names
        x = tweet.tokenize(x)          #tokenizer saving smiles
        x = [w.lower() for w in x if len(w)!=1]# and not re.match(r"[@&:;$~()<>`\[\]+#/\"*0-9%'.,!?\\-]" , w)]# del punct
        pos_tag = nltk.pos_tag(x)  #get list of word tags like verb
        for j, w in enumerate(x):
            x[j] = lemmat.lemmatize(w, tag_dict.get(pos_tag[j][1][0].upper(), wordnet.NOUN)) #set words to normal form
        x = [w for w in x if w not in stop_words] #del stopwords
        X[i]=[repeat_replacer.replace(w) for w in x]
    return X


def gen_wrd2idx(X):
    wrd2id={}
    for i, sent in enumerate(X):
        for j, token in enumerate(X[i]):
            if token not in wrd2id:
                wrd2id[token] = 1
            else:
                wrd2id[token]+=1  #word freq

    wrd2id={k: v for k, v in sorted(wrd2id.items(), key=lambda item: item[1], reverse=True)} #sort words by freq
    wrd2id={k: i+1 for i, k in enumerate(wrd2id.keys())} #set indxs
    wrd2id['-UNKNWN-']=0
    return wrd2id


def indexing(X, wrd2idx):
    for i, sent in enumerate(X):

        try:
            X[i]=sent.split(' ') #split tokens
        except Exception:
            X[i]=[0] #if empty sent

        for j, word in enumerate(X[i]):
            if word in wrd2idx:
                X[i][j]=wrd2idx[word] #set to tokens id
            else:
                X[i][j]=0 #incnwn word
        X[i]=np.array(X[i])
    X=np.array(X)
    return X



def get_dataset(path,word2idx_path,len_of_dir=10000,cut=500):
    dataframe = pd.read_csv(path, header=None, sep=',', encoding='utf-8')
    with open(word2idx_path, 'r') as json_file:
        wrd2idx = json.load(json_file)

    wrd2idx = dict(list(wrd2idx.items())[:len_of_dir]) #use len_of_dir most popular words
    wrd2idx['UNKNWN']=0
    dataset=dataframe.values
    X=dataset[:,0]
    Y=dataset[:,1]

    X=indexing(X,wrd2idx)

    for i, label in enumerate(Y):
        if label == 0:
            Y[i]=0
        elif label == 4:
            Y[i]=1

    Y=np.array(Y)
    X=sequence.pad_sequences(X,maxlen=cut) #set to equal sent size
    return (X,Y), wrd2idx

def preprocess(path,pref):
    size_of_dataset=1000000
    dataframe = pd.read_csv(path, header=None, sep=',', encoding='latin-1')
    dataset = dataframe.values
    X = dataset[:, 5]
    Y = dataset[:, 0]
    rand = list(range(len(dataset))) #shuffle data
    random.shuffle(rand)
    X = X[rand]
    Y = Y[rand]
    X = X[:size_of_dataset]
    Y = Y[:size_of_dataset]
    pd.DataFrame({'col1': X, "col2": Y}).to_csv(pref+'old_data.csv', header=False, index=False) #save to csv before preproc

    X=gen_tokens(X)
    wrd2idx=gen_wrd2idx(X)

    for i, sent in enumerate(X):
        string=' '.join([word for word in sent])
        X[i]=string

    pd.DataFrame({'col1': X, "col2": Y}).to_csv(pref+'data.csv', header=False, index=False,encoding='utf-8') #save ro csv after preproc

    return wrd2idx

if __name__ == "__main__":
    with open(word2idx_path,'w',encoding='utf-8') as out_file:
        json.dump(preprocess("data/training.csv",'train_'),out_file)



