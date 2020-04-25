from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from utils import save_sklearn_model

from deliverable.run_model import load_data
from deliverable.run_model import transform

if __name__ == "__main__":
    X, y = load_data("../data/data.npz")

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, test_size=0.15)

    model = make_pipeline(FunctionTransformer(transform), LinearRegression())

    model.fit(X_train, y_train)
    save_sklearn_model(model, "../deliverable/task1.pickle")
