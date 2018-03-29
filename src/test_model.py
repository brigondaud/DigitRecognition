#!/user/7/pepinau/2A/SIRR/virtualenv/py-sirr/bin/python

import sys
import random
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt


def prepare(dataset):
    """
    Reshaping, converting and normalizing data values.
    :param dataset: The set of training values
    """
    dataset = dataset.reshape(dataset.shape[0], 1, 28, 28)
    dataset = dataset.astype('float32')
    dataset /= 255
    return dataset


def main():
    if len(sys.argv) < 2:
        print("Missing one argument: neural network model (.h5 file)")
        sys.exit(1)

    # Loading the model and the MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    model = load_model(sys.argv[1])

    # Shaping test data
    X_test = prepare(X_test)

    predictions = model.predict_classes(X_test)
    bad_predictions = [i for i, (p, y) in enumerate(zip(predictions, y_test)) if p != y]

    # Plot 16 random incorrect predictions
    random.shuffle(bad_predictions)

    for i, c in enumerate(bad_predictions[:min([16, len(bad_predictions)])]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[c].reshape(28,28), interpolation='none')
        plt.title("Not {} but {}".format(predictions[c], y_test[c]))
        plt.xticks([])
        plt.yticks([])

    plt.show()


if __name__ == "__main__":
    main()
