from tensorflow.keras.datasets import cifar10

import numpy as np


def load_cifar10(num_classes=3):
    """
    Downloads CIFAR-10 dataset, which already contains a training and test set,
    and return the first `num_classes` classes.
    Example of usage:

    >>> (x_train, y_train), (x_test, y_test) = load_cifar10()

    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered data.
    """
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

    fil_train = y_train_all[:, 0] < num_classes
    fil_test = y_test_all[:, 0] < num_classes

    y_train = y_train_all[fil_train]
    y_test = y_test_all[fil_test]

    x_train = x_train_all[fil_train]
    x_test = x_test_all[fil_test]

    return (x_train, y_train), (x_test, y_test)


def load_cifar10_test(num_classes=3):
    """
    Used for loading only the test data of the CIFAR-10 dataset, for the first `num_classes` classes.
    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered test data.
    """
    (_, _), (x_test_all, y_test_all) = cifar10.load_data()
    fil_test = y_test_all[:, 0] < num_classes
    y_test = y_test_all[fil_test]
    x_test = x_test_all[fil_test]
    return x_test, y_test


def load_cifar10_train(num_classes=3):
    """
    Used for loading only the training data of the CIFAR-10 dataset, for the first `num_classes` classes.
    :param num_classes: int, default is 3 as required by the assignment.
    :return: the filtered training data.
    """
    (x_train_all, y_train_all), (_, _) = cifar10.load_data()
    fil_train = y_train_all[:, 0] < num_classes
    y_train = y_train_all[fil_train]
    x_train = x_train_all[fil_train]
    return x_train, y_train


def calculate_mce(predict, test):
    """
    Used for calculating the misclassification error, by comparing test data to prediction data.
    :param predict: array_like, predictions data set.
    :param test: array_like, test data set.
    :return: float, the misclassification error, such that: mce=[0,1].
    """
    assert test.shape == predict.shape
    return (np.argmax(predict, axis=1) != np.argmax(test, axis=1)).mean()
