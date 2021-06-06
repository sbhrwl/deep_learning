import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_model_layer_details(model):
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


def save_model_path(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_name_h5 = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")
    model_path = os.path.join(model_dir, file_name_h5)
    print(f"your model will be saved at the following location\n{model_path}")
    return model_path
