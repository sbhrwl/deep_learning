import time
import os
import tensorflow as tf
from tensorflow.keras import regularizers
# from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from src.core.common_utils import get_parameters


def get_learning_parameters():
    # Learning setup
    config = get_parameters()
    model_learning_setup = config["model_learning_setup"]
    model_learning_rate = model_learning_setup["learning_rate"]

    if model_learning_setup["optimizer"] == "sgd":
        sgd_momentum = model_learning_setup["momentum"]
        sgd_nesterov = model_learning_setup["nesterov"]

        optimizer = tf.keras.optimizers.SGD(learning_rate=model_learning_rate,
                                            momentum=sgd_momentum,
                                            nesterov=sgd_nesterov,
                                            name='SGD'
                                            )
    elif model_learning_setup["optimizer"] == "adagrad":
        initial_accumulator_value = model_learning_setup["initial_accumulator_value"]
        ada_grad_epsilon = model_learning_setup["epsilon"]

        optimizer = tf.keras.optimizers.Adagrad(learning_rate=model_learning_rate,
                                                initial_accumulator_value=initial_accumulator_value,
                                                epsilon=ada_grad_epsilon)
    elif model_learning_setup["optimizer"] == "adadelta":
        rho = model_learning_setup["rho"]
        ada_delta_epsilon = model_learning_setup["epsilon"]

        optimizer = tf.keras.optimizers.Adadelta(learning_rate=model_learning_rate,
                                                 rho=rho,
                                                 epsilon=ada_delta_epsilon)
    elif model_learning_setup["optimizer"] == "rmsprop":
        momentum = model_learning_setup["momentum"]
        rho = model_learning_setup["rho"]
        rms_prop_epsilon = model_learning_setup["epsilon"]

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=model_learning_rate,
                                                rho=rho,
                                                momentum=momentum,
                                                epsilon=rms_prop_epsilon)
    else:  # Consider other cases as adam
        adam_beta_1 = model_learning_setup["beta_1"]
        adam_beta_2 = model_learning_setup["beta_2"]
        adam_epsilon = model_learning_setup["epsilon"]

        optimizer = tf.keras.optimizers.Adam(learning_rate=model_learning_rate,
                                             beta_1=adam_beta_1,
                                             beta_2=adam_beta_2,
                                             epsilon=adam_epsilon)

    loss_function = model_learning_setup["loss_function"]
    metrics = model_learning_setup["metrics"]
    return loss_function, optimizer, metrics


def get_basic_model():
    config = get_parameters()

    configure_layers = config["model_training_parameters"]["configure_layers"]
    hidden_layer_1_name = configure_layers["hidden_layer_1_name"]
    hidden_layer_1_activation = configure_layers["hidden_layer_1_activation"]
    hidden_layer_1_number_of_neurons = configure_layers["hidden_layer_1_number_of_neurons"]
    hidden_layer_1_kernel_initializer = configure_layers["hidden_layer_1_kernel_initializer"]
    hidden_layer_2_name = configure_layers["hidden_layer_2_name"]
    hidden_layer_2_activation = configure_layers["hidden_layer_2_activation"]
    hidden_layer_2_number_of_neurons = configure_layers["hidden_layer_2_number_of_neurons"]
    hidden_layer_2_kernel_initializer = configure_layers["hidden_layer_2_kernel_initializer"]
    output_layer_name = configure_layers["output_layer_name"]
    output_layer_activation = configure_layers["output_layer_activation"]
    output_layer_number_of_neurons = configure_layers["output_layer_number_of_neurons"]

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"),
              tf.keras.layers.Dense(hidden_layer_1_number_of_neurons,
                                    activation=hidden_layer_1_activation,
                                    kernel_initializer=hidden_layer_1_kernel_initializer,
                                    # kernel_regularizer=regularizers.l1(l1=0.0001),
                                    # kernel_regularizer=regularizers.l2(l2=0.0001),
                                    # kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001),
                                    name=hidden_layer_1_name),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(hidden_layer_2_number_of_neurons,
                                    activation=hidden_layer_2_activation,
                                    kernel_initializer=hidden_layer_2_kernel_initializer,
                                    # kernel_regularizer=regularizers.l1(l1=0.0001),
                                    # kernel_regularizer=regularizers.l2(l2=0.0001),
                                    # kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001),
                                    name=hidden_layer_2_name),
              tf.keras.layers.Dropout(0.3),
              tf.keras.layers.Dense(output_layer_number_of_neurons,
                                    activation=output_layer_activation,
                                    name=output_layer_name)]

    tf_model = tf.keras.models.Sequential(LAYERS)
    print(tf_model.layers)
    print(tf_model.summary())

    loss_function, optimizer, metrics = get_learning_parameters()

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
    learning_curve_plot = ann_mnist_config["learning_curve_plot"]

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
    # If during fit loss values are "nan", then scale the dataset
    # StandardScaler.fit_transform

    # Step 3: Plot Learning curve
    plt.scatter(x=history.epoch, y=history.history['loss'], label='Training Error')
    plt.scatter(x=history.epoch, y=history.history['val_loss'], label='Validation Error')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.savefig(learning_curve_plot)
    # Short cut
    # pd.DataFrame(history.history).plot()


def save_model_path(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_name_h5 = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")
    model_path = os.path.join(model_dir, file_name_h5)
    print(f"your model will be saved at the following location\n{model_path}")
    return model_path


def get_bn_model():
    config = get_parameters()

    configure_layers = config["model_training_parameters"]["configure_layers"]
    hidden_layer_1_name = configure_layers["hidden_layer_1_name"]
    hidden_layer_1_activation = configure_layers["hidden_layer_1_activation"]
    hidden_layer_1_number_of_neurons = configure_layers["hidden_layer_1_number_of_neurons"]
    hidden_layer_1_kernel_initializer = configure_layers["hidden_layer_1_kernel_initializer"]
    hidden_layer_2_name = configure_layers["hidden_layer_2_name"]
    hidden_layer_2_activation = configure_layers["hidden_layer_2_activation"]
    hidden_layer_2_number_of_neurons = configure_layers["hidden_layer_2_number_of_neurons"]
    hidden_layer_2_kernel_initializer = configure_layers["hidden_layer_2_kernel_initializer"]
    output_layer_name = configure_layers["output_layer_name"]
    output_layer_activation = configure_layers["output_layer_activation"]
    output_layer_number_of_neurons = configure_layers["output_layer_number_of_neurons"]

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(hidden_layer_1_number_of_neurons,
                                    activation=hidden_layer_1_activation,
                                    kernel_initializer=hidden_layer_1_kernel_initializer,
                                    name=hidden_layer_1_name),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(hidden_layer_2_number_of_neurons,
                                    activation=hidden_layer_2_activation,
                                    kernel_initializer=hidden_layer_2_kernel_initializer,
                                    name=hidden_layer_2_name),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(output_layer_number_of_neurons,
                                    activation=output_layer_activation,
                                    name=output_layer_name)]

    tf_model = tf.keras.models.Sequential(LAYERS)
    print(tf_model.layers)
    print(tf_model.summary())

    loss_function, optimizer, metrics = get_learning_parameters()

    tf_model.compile(loss=loss_function,
                     optimizer=optimizer,
                     metrics=metrics)
    return tf_model


def get_bn_before_activation_function_model():
    config = get_parameters()

    configure_layers = config["model_training_parameters"]["configure_layers"]
    hidden_layer_1_name = configure_layers["hidden_layer_1_name"]
    hidden_layer_1_activation = configure_layers["hidden_layer_1_activation"]
    hidden_layer_1_number_of_neurons = configure_layers["hidden_layer_1_number_of_neurons"]
    hidden_layer_2_name = configure_layers["hidden_layer_2_name"]
    hidden_layer_2_activation = configure_layers["hidden_layer_2_activation"]
    hidden_layer_2_number_of_neurons = configure_layers["hidden_layer_2_number_of_neurons"]
    output_layer_name = configure_layers["output_layer_name"]
    output_layer_activation = configure_layers["output_layer_activation"]
    output_layer_number_of_neurons = configure_layers["output_layer_number_of_neurons"]

    LAYERS_BN_BIAS_FALSE = [
        tf.keras.layers.Flatten(input_shape=[28, 28], name="InputLayer"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_layer_1_number_of_neurons,
                              name=hidden_layer_1_name,
                              use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(hidden_layer_1_activation),
        tf.keras.layers.Dense(hidden_layer_2_number_of_neurons,
                              name=hidden_layer_2_name,
                              use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(hidden_layer_2_activation),
        tf.keras.layers.Dense(output_layer_number_of_neurons,
                              activation=output_layer_activation,
                              name=output_layer_name)]

    tf_model = tf.keras.models.Sequential(LAYERS_BN_BIAS_FALSE)
    print(tf_model.layers)
    print(tf_model.summary())

    loss_function, optimizer, metrics = get_learning_parameters()

    tf_model.compile(loss=loss_function,
                     optimizer=optimizer,
                     metrics=metrics)
    return tf_model


def get_model():
    config = get_parameters()
    model_type = config["model_learning_setup"]["model_type"]

    if model_type == "batch_normalisation":
        model = get_bn_model()
    elif model_type == "batch_normalisation_without_bias":
        model = get_bn_before_activation_function_model()
    else:
        model = get_basic_model()

    return model
