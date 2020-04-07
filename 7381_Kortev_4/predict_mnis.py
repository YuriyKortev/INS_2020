import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from sys import argv
from PIL import Image
import numpy as np


def load_img(path):
    im = Image.open(path).convert('L')
    im = im.resize((28, 28))
    im=np.array(im)
    im=255-im
    im=im/255
    im = np.expand_dims(im,axis=0)
    return im

assert argv[1]
sample=load_img(argv[1])

model=load_model('model.h5')
#print(sample)

print('На картинке цифра {}'.format(model.predict_classes(sample)[0]))