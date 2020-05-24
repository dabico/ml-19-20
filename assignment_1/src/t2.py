from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from utils import save_sklearn_model

from deliverable.run_model import load_data


def baseline_model():
    model = Sequential()
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(learning_rate=0.005))
    return model


if __name__ == "__main__":
    X, y = load_data("../data/data.npz")

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=1)

    model = KerasRegressor(build_fn=baseline_model, batch_size=40, epochs=500, verbose=0)

    model.fit(X_train, y_train)
    save_sklearn_model(model, "../deliverable/t2.pickle")
