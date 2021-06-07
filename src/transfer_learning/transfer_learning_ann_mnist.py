import tensorflow as tf
import sys

sys.path.append('./src')
from core.common_utils import get_parameters, get_data, get_scaled_train_validation_test_sets, basic_analysis
from core.model_utils import *


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    model_transfer_learning = config["model_transfer_learning"]
    model_to_load = model_transfer_learning["model_to_load"]

    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    CKPT_path = ann_mnist_config["checkpoint_path"]

    model_training_parameters = config["model_training_parameters"]
    epochs_to_train = model_training_parameters["epochs"]
    batch_size_for_training = model_training_parameters["batch"]

    # Step 1: Load data
    (X_train, y_train), (X_test, y_test) = get_data()

    # Step 2: Scale the train, validation and test set
    X_train, y_train, X_validation, y_validation, X_test = get_scaled_train_validation_test_sets(X_train,
                                                                                                 y_train,
                                                                                                 X_test)

    # Step 3: Load Previous model
    model = tf.keras.models.load_model(model_to_load)

    # Step 4: Check Model Details
    # 4.1: Summary
    print(model.summary())

    # 4.2: Trainable layers
    for layer in model.layers:
        print(f"{layer.name}: {layer.trainable}")

    # Step 5: Remove Last Layer
    for layer in model.layers[:-1]:
        layer.trainable = False
        print(f"{layer.name}: {layer.trainable}")

    # Step 6: Create new model
    lower_pretrained_layers = model.layers[:-1]

    new_model = tf.keras.models.Sequential(lower_pretrained_layers)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax")
        # tf.keras.layers.Dense(2, activation="binary")
    )
    print(new_model.summary())

    # Step 7: Setup Callbacks
    tb_cb, early_stopping_cb, check_pointing_cb = setup_callbacks_for_model_training(tensorboard_logs, CKPT_path)

    # # Step 8: Train
    # VALIDATION_SET = (X_validation, y_validation)
    # history = model.fit(X_train,
    #                     y_train,
    #                     epochs=epochs_to_train,
    #                     validation_data=VALIDATION_SET,
    #                     batch_size=batch_size_for_training,
    #                     callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])
    #
    # (loss, accuracy, val_loss, val_accuracy) = history.history
