import tensorflow as tf
import sys
# sys.path.insert(1, './src')
sys.path.append('./src')
from core.common_utils import get_parameters, get_data, get_scaled_train_validation_test_sets, basic_analysis
from core.model_utils import *


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    ann_mnist_config = config["ann_mnist_config"]
    tensorboard_logs = ann_mnist_config["tensorboard_logs"]
    CKPT_path = ann_mnist_config["checkpoint_path"]
    artifacts_dir = ann_mnist_config["artifacts_dir"]

    model_training_parameters = config["model_training_parameters"]
    epochs_to_train = model_training_parameters["epochs"]
    batch_size_for_training = model_training_parameters["batch"]

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

    # Step 6: Train
    VALIDATION_SET = (X_validation, y_validation)
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs_to_train,
                        validation_data=VALIDATION_SET,
                        batch_size=batch_size_for_training,
                        callbacks=[tb_cb, early_stopping_cb, check_pointing_cb])

    (loss, accuracy, val_loss, val_accuracy) = history.history

    UNIQUE_PATH = model.save(save_model_path(artifacts_dir))
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
