import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os


def layer_details(model):
    hidden_layer1 = model.layers[1]
    print("hidden_layer1.name", hidden_layer1.name)
    print(model.get_layer(hidden_layer1.name) is hidden_layer1)
    type(hidden_layer1.get_weights())
    hidden_layer1.get_weights()
    weights, biases = hidden_layer1.get_weights()
    print("shape of weights \n", weights.shape, "\n")
    print("shape of biases \n", biases.shape)


def get_log_path(log_dir="logs"):
    fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_dir, fileName)
    print(f"saving logs at: {log_path}")
    return log_path


def save_model_path(model_dir="mnist-model"):
    os.makedirs(model_dir, exist_ok=True)
    file_name_h5 = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")
    model_path = os.path.join(model_dir, file_name_h5)
    print(f"your model will be saved at the following location\n{model_path}")
    return model_path


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    print(f"data type of X_train_full: {X_train_full.dtype},\nshape of X_train_full: {X_train_full.shape}")

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # scale the test set as well
    X_test = X_test / 255.

    plt.imshow(X_train[0], cmap="binary")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(15, 15))
    sns.heatmap(X_train[0], annot=True, cmap="binary")
    # actual value of y_train
    y_train[0]

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
              tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
              tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
              tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)
    print(model_clf.layers)
    print(model_clf.summary())

    LOSS_FUNCTION = "sparse_categorical_crossentropy"  # use => tf.losses.sparse_categorical_crossentropy
    OPTIMIZER = "SGD"  # or use with custom learning rate=> tf.keras.optimizers.SGD(0.02)
    METRICS = ["accuracy"]

    model_clf.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=METRICS)

    EPOCHS = 10
    VALIDATION_SET = (X_valid, y_valid)

    log_dir = get_log_path()
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model_clf.fit(X_train,
                            y_train,
                            epochs=EPOCHS,
                            validation_data=VALIDATION_SET,
                            batch_size=32,
                            callbacks=[tb_cb])

    UNIQUE_PATH = model_clf.save(save_model_path())
    # loaded_model = tf.keras.models.load_model("<MODEL_NAME_WITH_LOCATION>")
    # Integrate MlFlow
    # https://drive.google.com/drive/u/3/folders/1HI4YeqqIXS4SxDr32YVqvhdFIQFkzk12

    # Jupyter NB
    # %load_ext tensorboard
    # %tensorboard --logdir logs

