import pandas as pd
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Flatten, Dense, Activation, Dropout
from sklearn.model_selection import GridSearchCV
from src.core.common_utils import get_parameters


def ann_hyper_parameter_tuning(X_train, y_train):
    # A. Define a create_model function, used by KerasClassifier constructor
    # Input parameters: (1)No of Layers, (2)Which Activation function to use
    def create_model(layers, activation):
        # Create Model
        model = Sequential()
        for i, nodes in enumerate(layers):
            # 1. Hidden layers
            if i == 0:
                # i. For Input layer, specify Input_dim as Number of features in the X_train
                model.add(Flatten(input_shape=[28, 28]))
                # ii. Specify Activation function
                model.add(Activation(activation))
                # iii. Specify Dropout or p value
                model.add(Dropout(0.3))
            else:
                model.add(Dense(nodes))
                model.add(Activation(activation))
                model.add(Dropout(0.3))

        model.add(Dense(units=10,
                        kernel_initializer='glorot_uniform',
                        activation='softmax'))

        # 2. Compiling the ANN
        configuration = get_parameters()
        model_metrics = configuration["model_metrics"]
        loss_function = model_metrics["loss_function"]
        optimizer = model_metrics["optimizer"]
        metrics = model_metrics["metrics"]

        # i. Which optimizer to use: GD/SGD/MBSGD/AdaGrad/RMSPROP: Adam is the best and most popular optimizer
        # ii. Which Loss function to use: binary_crossentropy as our problem is binary classification problem.
        # iii. Which metric to use
        model.compile(loss=loss_function,
                      optimizer=optimizer,
                      metrics=metrics)
        # print(model.layers, file=open('hyper_parameter_tuning_model_layers.txt', 'a'))
        # print(model.summary(), file=open('hyper_parameter_tuning_model_summary.txt', 'a'))
        return model

    config = get_parameters()
    estimator = config["hyper_parameter_tuning"]
    # number_of_hidden_layers = estimator["number_of_hidden_layers"]
    # number_of_hidden_layers = [(20), (40, 20), (45, 30, 15)]
    number_of_hidden_layers = [(40, 20)]
    print("number_of_hidden_layers", number_of_hidden_layers)
    activation_functions = estimator["activation_functions"]
    batch_sizes = estimator["batch_sizes"]
    number_of_epochs = estimator["number_of_epochs"]

    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = dict(layers=number_of_hidden_layers,
                      activation=activation_functions,
                      batch_size=batch_sizes,
                      epochs=number_of_epochs)

    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
    grid_result = grid.fit(X_train, y_train)
    pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    print(grid_result.best_score_, grid_result.best_params_)
