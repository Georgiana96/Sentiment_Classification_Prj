# import the necessary packages
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout
from keras.layers.core import Flatten
from keras.models import Sequential


class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=3, strides=1, input_shape=input_shape, activation='relu',
                                padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # after this the dimension is : 60x80

        # model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # after this the dimesion is : 30x40

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='sigmoid'))  # output layer

        return model


class VGG_16:
    @staticmethod
    def build(input_shape, classes):

        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=3, strides=1,input_shape=input_shape, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  #after this the dimension is : 60x80

        # model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2)) #after this the dimesion is : 30x40

        model.add(Convolution2D(filters=128, kernel_size=3, strides=1, activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2),strides=2)) #after this the dimesion is : 15x20

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='sigmoid')) #output layer

        return model
