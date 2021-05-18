import numpy as np
import pandas as pd
# import joblib

from perceptron_model import Perceptron


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

    # filename = 'AND_model.model'
    # joblib.dump(model, filename)