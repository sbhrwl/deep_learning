import tensorflow as tf
import mlflow.tensorflow
import sys

sys.path.append('./src')
from core.get_parameters import get_parameters
from core.utils import *


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    CKPT_path = ann_mnist_config["checkpoint_path"]

    model_training_parameters = config["model_training_parameters"]
    epochs_to_train = model_training_parameters["epochs"]
    batch_size_for_training = model_training_parameters["batch"]

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

    # Step 5: Setup Callbacks
    tb_cb, early_stopping_cb, check_pointing_cb = setup_callbacks_for_model_training(tensorboard_logs, CKPT_path)

    # Step 6: Setup MLFLOW
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    # if the value passed is 2,
    # mlflow will log the training metrics (loss, accuracy, and validation loss etc.) every 2 epochs.
    mlflow.tensorflow.autolog(every_n_iter=2)

    # Step 7: Train
    VALIDATION_SET = (X_validation, y_validation)
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs_to_train,
                        validation_data=VALIDATION_SET,
                        batch_size=batch_size_for_training,
                        callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

    (loss, accuracy, val_loss, val_accuracy) = history.history

    mlflow.end_run()
