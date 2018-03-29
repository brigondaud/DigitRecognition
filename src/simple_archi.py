#!/user/7/pepinau/2A/SIRR/virtualenv/py-sirr/bin/python

import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.callbacks import TensorBoard, EarlyStopping
import datetime

##################################################
# Intelligent systems: reasoning and recognition #
#      NEURAL NETWORK PROGRAMMING EXERCISE       #
##################################################
# Jose Munoz                                     #
# Aurelien Pepin                                 #
# Baptiste Rigondaud                             #
##################################################


def prepare(dataset, size):
    """
    Reshaping, converting and normalizing data values.
    :param dataset: The set of training values
    """
    dataset = dataset.reshape(size, 784)
    dataset = dataset.astype('float32')
    dataset /= 255
    return dataset

def simple(batchSize=32, ep=1, training=60000, test=10000):
    """
    Simple version to test loss function
    """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    losses=['mean_squared_error',
            'mean_absolute_error',
            'categorical_crossentropy',
            'logcosh']

    evals = []

    # Data preparation
    X_train = prepare(X_train[:training], training)
    X_test = prepare(X_test[:test], test)
    Y_train = np_utils.to_categorical(Y_train[:training], 10)      # 0..9
    Y_test = np_utils.to_categorical(Y_test[:test], 10)        # 0..9

    # --------------------
    # NEURAL NETWORK MODEL
    # --------------------

    # Model architecture
    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    for loss in losses:
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        tensorboard = TensorBoard(log_dir="logs_simple/{}".format("simple:"+loss))
        model.fit(X_train, Y_train, batch_size=batchSize, epochs=ep, verbose=1, callbacks=[tensorboard])

        evals.append((loss, model.evaluate(X_test, Y_test, verbose=1)))

    return evals


def main():
    K.set_image_dim_ordering('th')  # Tensorflow compatibility
    np.random.seed(123)             # For reproducibility
    print(simple(batchSize=50, ep=20))



if __name__ == "__main__":
    main()
