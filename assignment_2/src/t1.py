from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, LeakyReLU, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.utils import load_cifar10_train

import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    We first load the CIFAR-10 training data using a modification of the already provided load_cifar10 function.
    """
    (x_train, y_train) = load_cifar10_train()

    """
    Now we preprocess the data.
    First we have to normalize each pixel of each channel range to [0, 1].
    Then we create one-hot encoding for the labels using the to_categorical function.
    https://keras.io/api/utils/python_utils/#to_categorical-function
    """
    x_train = x_train / 255
    y_train = to_categorical(y_train, 3)

    """
    The validation set is formed from 20% of the training data
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    This will be necessary for implementing the early stopping later
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=30081998)

    """
    Next up is the definition for the neural network structure.
    We first add the convolution layer, with 8 filters of size 5 by 5, stride of 1 by 1, and ReLU activation.
    https://keras.io/api/layers/convolution_layers/convolution2d/
    Then we add a max pooling layer, with pooling size of 2 by 2.
    The size specification not necessary as it is assumed to be 2X2 by default.
    https://keras.io/api/layers/pooling_layers/max_pooling2d/
    Up next is the second convolution layer, with 16 filters of size 3 by 3, stride of 2 by 2, and ReLU activation.
    Now comes the average pooling layer, with pooling size of 2 by 2.
    Again, the size specification not necessary as it is 2X2 by default.
    https://keras.io/api/layers/pooling_layers/average_pooling2d/
    Then we add a layer to convert the 2D feature maps to flat vectors.
    https://keras.io/api/layers/reshaping_layers/flatten/
    Followed by two dropout layers with dropout probability of 0.3 and two dense layers.
    Each dense layer is preceded by a dropout layer.
    https://keras.io/api/layers/regularization_layers/dropout/
    https://keras.io/api/layers/core_layers/dense/
    The first dense layer has 8 neurons and tanh activation, and L2 regularization with a factor of 0.005,
    while the second dense output layer has 3 class neurons, softmax activation, and L2 regularization with a
    factor of 0.005.
    """
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='tanh', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax", kernel_regularizer=l2(0.005)))

    """
    Having defined the neural network, we now configure the model for training
    As specified by the assignment:
    - We use the RMSprop optimization algorithm, with a learning rate of 0.003
    - We use categorical cross-entropy as a loss function
    We also have to implement early stopping callback
    https://keras.io/api/callbacks/early_stopping/
    Again, as specified by the assignment:
    - The validation accuracy is used as the monitor for early stopping
    - It has a patience of 10 epochs
    Setting the mode to 'max' ensures that training will stop when the validation accuracy stops increasing.
    By enabling restore_best_weights, we ensure that the model weights from the epoch with the best value of
    validation accuracy are restored.
    """
    model.compile(optimizer=RMSprop(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=10, restore_best_weights=True, mode='max')

    """
    With the training and validation sets defined, as well as the network model and early stopping procedure set up,
    we can finally fit the data.
    According to the requirements:
    - The batch size is set to 128
    - Training is done for a max of 500 epochs
    - The Early Stopping is passed as a callback, to ensure that it is applied during training
    We store the fitting history callback into a variable, which will be used for the plot.
    """
    hist_cb = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=128, callbacks=[es],
                        verbose=0, epochs=500)

    """
    The following code defines the plot, its size, and then passes the accuracy values from history for plotting.
    Labels are added to the axes, the x axis representing the epochs, while the y axis represents the accuracy measures.
    After adjusting the margins and legend, the plot is then shown on screen.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(hist_cb.history['accuracy'], label='training accuracy')
    plt.plot(hist_cb.history['val_accuracy'], label='validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.margins(x=0, y=0)
    plt.legend()
    plt.show()

    """
    Finally, save the model to an h5 file in the deliverable directory
    """
    model.save('../deliverable/nn_task1.h5')
