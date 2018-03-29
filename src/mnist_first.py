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

##################################################
# Intelligent systems: reasoning and recognition #
#      NEURAL NETWORK PROGRAMMING EXERCISE       #
##################################################
# Jose Munoz                                     #
# Aurelien Pepin                                 #
# Baptiste Rigondaud                             #
##################################################

# TODO. Use save methods from Keras to avoid recomputing the whole network at each run.
# TODO. We can use command line parameters to either keep the model or recompute it.
# CURRENT ACCURACY (cf. model.evaluate): 98.4%

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

def mnist_v1():
    """
    Basic version of handwritten digits recognition.
    """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Data preparation
    X_train = prepare(X_train)
    X_test = prepare(X_test)
    Y_train = np_utils.to_categorical(Y_train, 10)      # 0..9
    Y_test = np_utils.to_categorical(Y_test, 10)        # 0..9

    # Fitting the data to the augmentation data generator
    datagen = augmentedData(X_train)

    # --------------------
    # NEURAL NETWORK MODEL
    # --------------------

    # Model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Model compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128), epochs=20, verbose=1)

    now = datetime.datetime.now()
    model.save("sirr_epochs20_" + str(now.hour) + "h" + str(now.minute) + ".h5")

    # Model evaluation
    return model.evaluate(X_test, Y_test, verbose=1)


def main():
    K.set_image_dim_ordering('th')  # Tensorflow compatibility
    np.random.seed(123)             # For reproducibility
    print(mnist_v1())


if __name__ == "__main__":
    main()
