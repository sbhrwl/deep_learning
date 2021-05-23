import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from urllib.parse import urlparse

import time
import os
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from get_parameters import get_parameters


def get_log_path(log_directory):
    file_name = time.strftime("log_%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_directory, file_name)
    print(f"saving logs at: {log_path}")
    return log_path


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    print(f"data type of X_train_full: {X_train_full.dtype},\nshape of X_train_full: {X_train_full.shape}")

    # scale the train, validation and test set
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.

    plt.imshow(X_train[0], cmap="binary")
    plt.axis('off')
    # plt.show()

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
    BATCH = 55
    VALIDATION_SET = (X_valid, y_valid)

    config = get_parameters()
    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]

    # Callbacks
    log_dir = get_log_path(tensorboard_logs)
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    CKPT_path = ann_mnist_config["checkpoint_path"]
    check_pointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    config = get_parameters()
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    registered_mlflow_model = mlflow_config["registered_model_name"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        history = model_clf.fit(X_train,
                                y_train,
                                epochs=EPOCHS,
                                validation_data=VALIDATION_SET,
                                batch_size=BATCH,
                                callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

        (loss, accuracy, val_loss, val_accuracy) = history.history

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH)

        # mlflow.log_metric("loss", loss)
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("val_loss", val_loss)
        # mlflow.log_metric("val_accuracy", val_accuracy)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        # if tracking_url_type_store != "file":
        #     mlflow.sklearn.log_model(
        #         model_clf,
        #         "model",
        #         registered_model_name=registered_mlflow_model)
        # else:
        #     mlflow.sklearn.load_model(model_clf, "model")
