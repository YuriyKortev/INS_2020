import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Input
from keras.models import Model
from matplotlib import pyplot as plt
import csv
import collections

np.random.seed(999)

def save_csv(name, data):
    file = open(name, "w+")
    my_csv = csv.writer(file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    if isinstance(data, collections.Iterable) and isinstance(data[0], collections.Iterable):
        for i in data:
            my_csv.writerow(i)
    else:
        my_csv.writerow(data)

def genDataset(len):
    dataset=[]
    labels=[]
    for i in range(len):
        x=np.random.normal(0,10)
        e=np.random.normal(0,0.3)
        sample=np.array([x**2+x+e,np.sin(x-np.pi/4)+e,np.log10(np.abs(x))+e,(-x)**3+e,(-x)/4+e,-x+e])
        dataset.append(sample)
        labels.append(np.abs(x)+e)

    return np.array(np.round(dataset,2)), np.array(np.round(labels,2))

def cr_model():
    main_input=Input(shape=(6,),name='main_input')
    encoded=Dense(16,activation='relu')(main_input)
    encoded=Dense(6, name='enc',activation='relu')(encoded)
    encoder=Model(main_input,encoded, name='encoder')

    dec_inp=Input(shape=(6,), name='inp_dec')
    decoded = Dense(16, activation='relu')(dec_inp)
    decoded = Dense(6, name='dec')(decoded)
    decoder=Model(dec_inp,decoded, name='decoder')


    regr=Dense(20,activation='relu')(encoded)
    regr=Dense(18,activation='relu')(regr)
    regr = Dense(1, name='regr')(regr)
    regr_mod=Model(main_input,regr,name='regres')


    model=Model(main_input,[decoder(encoder(main_input)), regr_mod(main_input)])

    model.compile(loss={
        'regres': 'mse',
        'decoder':'mse'
    }, optimizer='rmsprop', metrics={
        'regres': 'mae',
        'decoder':'mae'
    })


    return model, encoder,  decoder, regr_mod


main_model, encoder, decoder, regression = cr_model()
x, y =genDataset(400)

num_ep=300

h=main_model.fit(x, {
    'regres': y,
    'decoder':x
}, epochs=num_ep, batch_size=1, verbose=0, validation_split=0.2)

#print(encoder.predict(x),'\n\n',x)
#print(h.history)
plt.plot(range(num_ep),h.history['val_regres_mae'])
plt.show()
plt.plot(range(num_ep)[50:],h.history['val_decoder_mae'][50:],'r--')
plt.show()

regression.save('resres.h5')
encoder.save('encoder.h5')
decoder.save('decoder.h5')

save_csv('./x.csv', x)
save_csv('./y.csv', y)

save_csv('./encoded.csv', encoder.predict(x))
save_csv('./decoded.csv', decoder.predict(encoder.predict(x)))
save_csv('./regression.csv', regression.predict(x))



