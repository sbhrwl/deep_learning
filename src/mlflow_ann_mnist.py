import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.tensorflow

import time
import os
import sys

# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src/get_parameters')
from src.core.get_parameters import get_parameters


def get_log_path(log_directory):
    file_name = time.strftime("log_%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_directory, file_name)
    print(f"saving logs at: {log_path}")
    return log_path


def get_data():
    mnist = tf.keras.datasets.mnist
    (features_train, target_train), (features_test, target_test) = mnist.load_data()
    print(f"data type of features_train: {features_train.dtype},\nshape of features_train: {features_train.shape}")
    return (features_train, target_train), (features_test, target_test)


def get_scaled_train_validation_test_sets(features_train, target_train, features_test):
    features_validation = features_train[:5000] / 255.
    features_train = features_train[5000:] / 255.
    target_validation = target_train[:5000]
    target_train = target_train[5000:]
    features_test = features_test / 255.
    return features_train, target_train, features_validation, target_validation, features_test


def basic_analysis(features_train, target_train):
    plt.imshow(features_train[0], cmap="binary")
    plt.axis('off')
    # plt.show()

    plt.figure(figsize=(15, 15))
    sns.heatmap(features_train[0], annot=True, cmap="binary")
    # actual value of target_train
    # print(target_train[0])


def get_model():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"),
              tf.keras.layers.Dense(300, activation="relu", name="HiddenLayer1"),
              tf.keras.layers.Dense(100, activation="relu", name="HiddenLayer2"),
              tf.keras.layers.Dense(10, activation="softmax", name="OutputLayer")]

    tf_model = tf.keras.models.Sequential(LAYERS)
    print(tf_model.layers)
    print(tf_model.summary())

    # Set Metrics for the model
    LOSS_FUNCTION = "sparse_categorical_crossentropy"  # use => tf.losses.sparse_categorical_crossentropy
    OPTIMIZER = "SGD"  # or use with custom learning rate=> tf.keras.optimizers.SGD(0.02)
    METRICS = ["accuracy"]

    tf_model.compile(loss=LOSS_FUNCTION,
                     optimizer=OPTIMIZER,
                     metrics=METRICS)
    return tf_model


def setup_callbacks_for_model_training(model_tensorboard_logs, model_CKPT_path):
    log_dir = get_log_path(model_tensorboard_logs)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    check_pointing_callback = tf.keras.callbacks.ModelCheckpoint(model_CKPT_path, save_best_only=True)
    return tb_callback, early_stopping_callback, check_pointing_callback


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    CKPT_path = ann_mnist_config["checkpoint_path"]

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Step 1: Load data
    (X_train, y_train), (X_test, y_test) = get_data()

    # Step 2: Scale the train, validation and test set
    X_train, y_train, X_validation, y_validation, X_test = get_scaled_train_validation_test_sets(X_train,
                                                                                                 y_train,
                                                                                                 X_test)

    # Step 3: Analyse train data
    basic_analysis(X_train, y_train)

    # Step 4: Create Model
    model = get_model()

    # Step 5: Set Hyper parameters
    EPOCHS = 10
    BATCH = 55
    VALIDATION_SET = (X_validation, y_validation)

    # Step 6: Setup Callbacks
    tb_cb, early_stopping_cb, check_pointing_cb = setup_callbacks_for_model_training(tensorboard_logs, CKPT_path)

    # Step 7: Setup MLFLOW
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    # if the value passed is 2,
    # mlflow will log the training metrics (loss, accuracy, and validation loss etc.) every 2 epochs.
    mlflow.tensorflow.autolog(every_n_iter=2)

    # Step 8: Train
    history = model.fit(X_train,
                        y_train,
                        epochs=EPOCHS,
                        validation_data=VALIDATION_SET,
                        batch_size=BATCH,
                        callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

    (loss, accuracy, val_loss, val_accuracy) = history.history

    mlflow.end_run()
