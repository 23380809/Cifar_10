from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10, cifar100
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from keras.optimizers import SGD
from random import shuffle
import imageio
import numpy as np
from matplotlib import pyplot
from keras import optimizers
import sys
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib




class load_x_y():
    def one_hot_en(self, y):
        encode = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        encode[y] = 1.
        return encode

    def load_data(self, total_case):
        s, result, x_t, y_t = '', [], [], []
        options= [0,1,2,3,4,5,6,7,8,9]
        for i in range(10):
            s = s + str(options[i]) + '/'
            for j in range(0, total_case, 1):
                s += str('{0:04}'.format(j+1))
                result.append(s + '.png')
                s = s[:2]
            s = ''
        shuffle(result)

        for pic in result:
            y = self.one_hot_en(int(pic[:1]))
            y_t.append(y)
            im = imageio.imread(pic)
            im = im / 255
            x_t.append(im)
        x_t = np.array(x_t)
        y_t = np.array(y_t)
        return x_t, y_t

    def load_test_data(self, total_case):
        s, test = '', []
        for i in range(total_case):
            s += str('{0:04}'.format(i+1))
            im = imageio.imread('test/' + s + '.png')
            im = im /255
            test.append(im)
            s = ''
        test = np.array(test)
        return test


def one_hot_encoding(y):
    encode = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    encode[y] = 1.
    return encode



def load():
    choice = [33,57,85,25,44,35,56,13,43,79]
    buffer = []
    x = []
    y = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    (cifar_x, cifar_y), (cifar_test_x, cifar_test_y) = cifar100.load_data()
    cifar_x = cifar_x.reshape(cifar_x.shape[0], 32, 32, 3).astype('float32')
    cifar_test_x = cifar_test_x.reshape(cifar_test_x.shape[0], 32, 32, 3).astype('float32')
    cifar_x = cifar_x / 255.0
    cifar_test_x = cifar_test_x / 255.0

    cifar_y = np.array(cifar_y)
    for i in range(len(cifar_y)):
        buffer.append(cifar_y[i][0])
    for data, tag in zip(cifar_x, buffer):
        if tag in choice:
            x.append(data)
            y.append(tag)
    x_train = np.array(x)
    y = np.array(y)
    y = y / 10
    for i in y:
        y_train.append(one_hot_encoding(int(i)))

    x = []
    y = []
    buffer = []

    y_train = np.array(y_train)

    for i in range(len(cifar_test_y)):
        buffer.append(cifar_test_y[i][0])
    for data, tag in zip(cifar_test_x, buffer):
        if tag in choice:
            x.append(data)
            y.append(tag)
    x_test = np.array(x)
    y = np.array(y)
    y = y / 10
    for i in y:
        y_test.append(one_hot_encoding(int(i)))

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load()


l = load_x_y()


x_learn, y_learn = l.load_data(500)
x_t = l.load_test_data(500)


datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, zoom_range=0.1, validation_split=0.2)


train = datagen.flow(x_learn, y_learn, batch_size=64)



model = load_model('test.h5')
model.save('previous_model.h5')
model.summary()
sgd = SGD(learning_rate=0.08, decay=5e-4, momentum=0.9, nesterov=True)


model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, batch_size=64, epochs=10, validation_split=0.2)
#model.fit(it_train, batch_size=64, epochs=10)

prediction = model.predict(x_t)

ans = ''
answer = open('answer.txt', 'w+')
for predict in prediction:
    predict = list(predict)
    ans += str(predict.index(max(predict)))


for i in range(len(ans)):
    str1 = str('{0:04}'.format(i + 1))
    answer.write(str1 + ' ' + ans[i:i+1] + '\n')


model.evaluate(x_test, y_test, batch_size=32)

model.save('test.h5')
