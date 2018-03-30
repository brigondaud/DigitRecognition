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

# CURRENT ACCURACY (cf. model.evaluate): 99.41%
# Based on Yann Le Cun's paper (LeNet)


def prepare(dataset):
    """
    Reshaping, converting and normalizing data values.
    :param dataset: The set of training values
    """
    dataset = dataset.reshape(dataset.shape[0], 1, 28, 28)
    dataset = dataset.astype('float32')
    dataset /= 255
    return dataset


def augmentedData(trainningData):
    """
    Data augmentation for training data.
    """
    datagen = ImageDataGenerator (
        width_shift_range=0.075,
        height_shift_range=0.075,
        rotation_range=12,
        shear_range=0.075,
        zoom_range=0.05,
        fill_mode='constant',
        cval=0
    )

    datagen.fit(trainningData)
    return datagen


def convolution(batchSize=32, ep=1, training=60000, test=10000):
    """
    Basic version of handwritten digits recognition.
    """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Data preparation
    X_train = prepare(X_train[:training])
    X_test = prepare(X_test[:test])
    Y_train = np_utils.to_categorical(Y_train[:training], 10)      # 0..9
    Y_test = np_utils.to_categorical(Y_test[:test], 10)        # 0..9

    # Fitting the data to the augmentation data generator
    datagen = augmentedData(X_train)

    # --------------------
    # NEURAL NETWORK MODEL
    # --------------------

    # Model architecture
    model = Sequential()

    # 4 filters per image: but takes too long
    model.add(Conv2D(filters=batchSize, kernel_size=(4, 4), activation='relu', input_shape=(1, 28, 28)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 6 filters per image: but takes too long
    model.add(Conv2D(filters=batchSize, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(6 * batchSize, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Model compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Tensor board saves and early stopping callbacks
    now = datetime.datetime.now()
    tensorboard = TensorBoard(log_dir="logs/{}".format(str(now.hour) +":"+str(now.minute)))
    early = EarlyStopping(monitor='loss', patience=2)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batchSize), epochs=ep, verbose=1, callbacks=[tensorboard, early])

    # Model saves
    now = datetime.datetime.now()
    model.save("models/sirr_mnist_archi2_" + str(now.hour) + "h" + str(now.minute) + ".h5")

    # Model evaluation
    return model.evaluate(X_test, Y_test, verbose=1)


def main():
    K.set_image_dim_ordering('th')  # Tensorflow compatibility
    np.random.seed(123)             # For reproducibility
    print(convolution(batchSize=50, ep=20))
    print("\n\nFor Tensorboard, execute: tensorboard --logdir=logs/\n\n")


if __name__ == "__main__":
    main()
