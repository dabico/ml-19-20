import joblib
import numpy as np


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates the mean squared error between the values in y_true and the values
    in y_pred.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param y_true: Numpy array, the true target values from the test set;
    :param y_pred: Numpy array, the values predicted by your model.
    :return: float, the the mean squared error between the two arrays.
    """
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred) ** 2).mean()


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


def transform(x):
    return np.hstack((x, (np.sin(x[:, 0]) * x[:, 1]).reshape(-1, 1)))


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    x, y = load_data('../data/data.npz')

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the trained model
    baseline_model = load_model('./baseline_model.pickle')
    task1_model = load_model('./t1.pickle')
    task2_model = load_model('./t2.pickle')

    # Predict on the given samples
    # Uncomment as needed
    y_pred_bline = baseline_model.predict(x)
    y_pred_task1 = task1_model.predict(x)
    y_pred_task2 = task2_model.predict(x)

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse_bline = evaluate_predictions(y_pred_bline, y)
    mse_task1 = evaluate_predictions(y_pred_task1, y)
    mse_task2 = evaluate_predictions(y_pred_task2, y)
    print('Baseline MSE: {}'.format(mse_bline))
    print('Task 1   MSE: {}'.format(mse_task1))
    print('Task 2   MSE: {}'.format(mse_task2))
