from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from src.utils import load_cifar10_test, calculate_mce

if __name__ == '__main__':
    """
    First load the CIFAR-10 test data, and preprocess it.
    """
    (x_test, y_test) = load_cifar10_test()
    x_test = x_test / 255
    y_test = to_categorical(y_test, 3)

    """
    Then load the trained models and predict on the given samples.
    """
    model_task1 = load_model('./nn_task1.h5')
    model_task2 = load_model('./nn_task2.h5')
    y_pred_task1 = model_task1.predict(x_test)
    y_pred_task2 = model_task2.predict(x_test)

    """
    Finally evaluate the misclassification error for the two models.
    The misclassification error of the two models is:
    - task 1: 0.1493...
    - task 2: 0.1546...
    The difference between the two being: 0.0053...
    """
    mce1 = calculate_mce(y_pred_task1, y_test)
    mce2 = calculate_mce(y_pred_task2, y_test)
    print("Misclassification error task 1:", mce1)
    print("Misclassification error task 2:", mce2)
    print("Difference: ", abs(mce1 - mce2))
