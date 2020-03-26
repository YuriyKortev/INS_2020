import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Sequential
from keras.layers import Dense


def relu(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tens(inp, weights):
    inp=inp.copy()
    for i in range(len(weights)-1):
        inp=relu(np.dot(inp,weights[i][0])+weights[i][1])
    inp = sigmoid(np.dot(inp,weights[i+1][0]) + weights[i+1][1])
    return inp


def naive(data, weights):
    layer = [relu for _ in range(len(weights) - 1)]
    layer.append(sigmoid)
    data = data.copy()
    for l in range(len(weights)):
        res = np.zeros((data.shape[0], weights[l][0].shape[1]))
        for i in range(data.shape[0]):
            for j in range(weights[l][0].shape[1]):
                sum = 0
                for k in range(data.shape[1]):
                    sum += data[i][k] * weights[l][0][k][j]
                res[i][j] = layer[l](sum + weights[l][1][j])
        data = res
    return data

def log(a,b):
    return (a or b) and (a != (not b))

def compr(inp, mod):
    weights = [i.get_weights() for i in mod.layers]
    print('model res: ')
    print(model.predict(inp))
    print('naive res: ')
    print(naive(inp,weights))
    print('tensor res: ')
    print(tens(inp,weights))

inp = np.genfromtxt('tensor', delimiter=';')


labels= np.array([log(i[0],i[1]) for i in inp])

model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(2,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

print('true answer: \n',np.array([log(*x) for x in inp]))

compr(inp,model)
print('\nfitting...\n')
model.fit(inp, np.array([log(*x) for x in inp]),epochs=200,batch_size=1,verbose=0)
compr(inp, model)

print('true answer: \n',np.array([log(*x) for x in inp]))