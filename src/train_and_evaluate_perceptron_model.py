import numpy as np
import pandas as pd
import joblib

from perceptron_model import Perceptron
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(self.X), 1))]
    return self.thatsWhatPerceptronDoes(X_with_bias, self.weights)


if __name__ == "__main__":
    data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 0, 0, 1]}

    AND = pd.DataFrame(data)
    AND

    X = AND.drop("y", axis=1)
    X

    y = AND['y']
    y.to_frame()

    # model = Perceptron(eta = 0.5, epochs=10, thatsWhatPerceptronDoes=thatsWhatPerceptronDoes)
    model = Perceptron(eta=0.5, epochs=2)
    model.fit(X, y)
    # model.predict(X)

    config = get_parameters()
    perceptron_config = config["perceptron_config"]
    filename = perceptron_config["artifacts_dir"]
    joblib.dump(model, filename)
