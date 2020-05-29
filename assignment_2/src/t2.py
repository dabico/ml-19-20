from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, AveragePooling2D, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from itertools import product
import numpy as np

from src.utils import load_cifar10_train

if __name__ == "__main__":
    """
    Similarly to T1, we first load the CIFAR-10 training data, then we preprocess it, and finally we extract the
    validation set from 20% of the training data.
    """
    (x_train, y_train) = load_cifar10_train()
    x_train = x_train / 255
    y_train = to_categorical(y_train, 3)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=30081998)

    """
    To tune the learning rate, we first have to define the hyper-parameter tuples from the two given parameter ranges.
    Neurons is an array containing all valid neuron number values for the last dense layer.
    The range goes from [8,64], and encompasses only multiples of 8.
    Rates is an array containing all valid learning rate values for the optimizer.
    This array only encompasses only three values.
    Once all the hyper-parameter range values are defined, we construct an array of parameter tuples using the product
    function of itertools, which mimics the use of a Cartesian product.
    https://stackoverflow.com/a/21652010
    """
    neurons = [8, 16, 24, 32, 40, 48, 56, 64]
    rates = [0.01, 0.001, 0.0001]
    parameters = list(product(neurons, rates))

    """
    We then iterate over the hyper-parameter tuple array to evaluate each of the generated parameter combinations with
    values from the validation set.
    The network setup mimics that of the one found in Task 1, except we swap the number of neurons in the dense layer
    and the RMSprop learning rate with the respective hyper-parameters.
    Evaluation results are appended to the evaluations array.
    
    Keep in mind that evaluating 24 models takes approximately two and a half hours...
    """
    evaluations = []
    for (neuron, rate) in parameters:
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(neuron, activation='tanh', kernel_regularizer=l2(0.005)))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation="softmax", kernel_regularizer=l2(0.005)))
        model.compile(optimizer=RMSprop(learning_rate=rate), loss='categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=10, restore_best_weights=True, mode='max')
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                  batch_size=128, callbacks=[es], verbose=0, epochs=500)
        _, accuracy = model.evaluate(x_valid, y_valid)
        evaluations.append(accuracy)

    """
    Having evaluated all the hyper-parameter configurations, we take the best one to fit the data.
    The best configuration found involved 16 neurons and a learning rate of 0.001.
    """
    (neuron_best, rate_best) = parameters[np.argmax(evaluations)]
    print("Neurons: ", neuron_best)
    print("Learning Rate: ", rate_best)

    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(neuron_best, activation='tanh', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax", kernel_regularizer=l2(0.005)))
    model.compile(optimizer=RMSprop(learning_rate=rate_best), loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=10, restore_best_weights=True, mode='max')
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
              batch_size=128, callbacks=[es], verbose=0, epochs=500)

    """
    Finally, save the model to an h5 file in the deliverable directory
    """
    model.save('../deliverable/nn_task2.h5')
