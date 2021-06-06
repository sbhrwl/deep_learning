import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
import sys
# sys.path.insert(1, './src/get_parameters')
sys.path.append('./src')
from core.get_parameters import get_parameters


def layer_details(model):
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


def save_model_path(model_dir):
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
    VALIDATION_SET = (X_valid, y_valid)

    config = get_parameters()
    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    artifacts_dir = ann_mnist_config["artifacts_dir"]

    # Callbacks
    log_dir = get_log_path(tensorboard_logs)
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    CKPT_path = ann_mnist_config["checkpoint_path"]
    check_pointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    history = model_clf.fit(X_train,
                            y_train,
                            epochs=EPOCHS,
                            validation_data=VALIDATION_SET,
                            batch_size=32,
                            callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

    UNIQUE_PATH = model_clf.save(save_model_path(artifacts_dir))
    # loaded_model = tf.keras.models.load_model("<MODEL_NAME_WITH_LOCATION>")

    # Jupyter NB
    # %load_ext tensorboard
    # %tensorboard --logdir logs

    # load model from CKPT
    # ckpt_model = tf.keras.models.load_model(CKPT_path)
    #
    # history = ckpt_model.fit(X_train,
    #                          y_train,
    #                          epochs=EPOCHS,
    #                          validation_data=VALIDATION_SET,
    #                          batch_size=32,
    #                          callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])
