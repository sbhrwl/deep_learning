import numpy as np
import sys

sys.path.append('./src')
from core.common_utils import get_parameters, get_data, get_scaled_train_validation_test_sets, basic_analysis
from core.model_utils import *
from transfer_learning.transfer_learning_ann_mnist import get_model_via_transfer_learning


def update_even_odd_labels(labels):
    for idx, label in enumerate(labels):
        labels[idx] = np.where(label % 2 == 0, 1, 0)
    return labels


if __name__ == "__main__":
    config = get_parameters()

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
    # Step 3: Format Target Values
    y_train_binary, y_test_binary, y_validation_binary = update_even_odd_labels([y_train, y_test, y_validation])
    print(np.unique(y_validation_binary))

    # Step 4: Get Model Via transfer learning
    model = get_model_via_transfer_learning()
    print("New_model Summary")
    print(model.summary())

    # Step 7: Setup Callbacks
    tb_cb, early_stopping_cb, check_pointing_cb = setup_callbacks_for_model_training(tensorboard_logs, CKPT_path)

    # Step 8: Train
    VALIDATION_SET = (X_validation, y_validation_binary)

    history = model.fit(X_train,
                        y_train_binary,
                        epochs=epochs_to_train,
                        validation_data=VALIDATION_SET,
                        batch_size=batch_size_for_training,
                        callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

    (loss, accuracy, val_loss, val_accuracy) = history.history

    # Step 9: Evaluate
    model.evaluate(X_test, y_test_binary)

    # 9.1: Evaluation Test
    X_new = X_test[:3]
    y_test[:3], y_test_binary[:3]

    # 9.2: Verify Predictions
    predictions = np.argmax(model.predict(X_new), axis=-1)
    # model.predict(X_new)
    print(predictions)
