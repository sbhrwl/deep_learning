import time
import os
import tensorflow as tf
import sys

sys.path.append('./src')
from core.common_utils import get_parameters


def get_model():
    config = get_parameters()

    configure_layers = config["model_training_parameters"]["configure_layers"]
    hidden_layer_1_name = configure_layers["hidden_layer_1_name"]
    hidden_layer_1_activation = configure_layers["hidden_layer_1_activation"]
    hidden_layer_2_name = configure_layers["hidden_layer_2_name"]
    hidden_layer_2_activation = configure_layers["hidden_layer_2_activation"]
    output_layer_name = configure_layers["output_layer_name"]
    output_layer_activation = configure_layers["output_layer_activation"]

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"),
              tf.keras.layers.Dense(300, activation=hidden_layer_1_activation, name=hidden_layer_1_name),
              tf.keras.layers.Dense(100, activation=hidden_layer_2_activation, name=hidden_layer_2_name),
              tf.keras.layers.Dense(10, activation=output_layer_activation, name=output_layer_name)]

    tf_model = tf.keras.models.Sequential(LAYERS)
    print(tf_model.layers)
    print(tf_model.summary())

    config = get_parameters()

    model_metrics = config["model_metrics"]
    loss_function = model_metrics["loss_function"]
    optimizer = model_metrics["optimizer"]
    metrics = model_metrics["metrics"]

    tf_model.compile(loss=loss_function,
                     optimizer=optimizer,
                     metrics=metrics)
    return tf_model


def get_model_layer_details(model):
    print(model.summary())
    hidden_layer1 = model.layers[1]
    print("hidden_layer1.name", hidden_layer1.name)
    print(model.get_layer(hidden_layer1.name) is hidden_layer1)
    type(hidden_layer1.get_weights())
    hidden_layer1.get_weights()
    weights, biases = hidden_layer1.get_weights()
    print("shape of weights \n", weights.shape, "\n")
    print("shape of biases \n", biases.shape)


def get_log_path(log_directory):
    file_name = time.strftime("log_%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_directory, file_name)
    print(f"saving logs at: {log_path}")
    return log_path


def setup_callbacks_for_model_training(model_tensorboard_logs, model_CKPT_path):
    log_dir = get_log_path(model_tensorboard_logs)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    check_pointing_callback = tf.keras.callbacks.ModelCheckpoint(model_CKPT_path, save_best_only=True)
    return tb_callback, early_stopping_callback, check_pointing_callback


def train_model(model_to_train, train_features, train_target, validation_data):
    config = get_parameters()

    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    CKPT_path = ann_mnist_config["checkpoint_path"]

    model_training_parameters = config["model_training_parameters"]
    epochs_to_train = model_training_parameters["epochs"]
    batch_size_for_training = model_training_parameters["batch"]

    # Step 1: Setup Callbacks
    tb_cb, early_stopping_cb, check_pointing_cb = setup_callbacks_for_model_training(tensorboard_logs, CKPT_path)

    # Step 2: Train
    history = model_to_train.fit(train_features,
                                 train_target,
                                 epochs=epochs_to_train,
                                 validation_data=validation_data,
                                 batch_size=batch_size_for_training,
                                 callbacks=[tb_cb, early_stopping_cb, check_pointing_cb],
                                 verbose=2)

    # (loss, accuracy, val_loss, val_accuracy) = history.history


def save_model_path(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_name_h5 = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")
    model_path = os.path.join(model_dir, file_name_h5)
    print(f"your model will be saved at the following location\n{model_path}")
    return model_path
