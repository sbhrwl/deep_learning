import tensorflow as tf
import sys
# sys.path.insert(1, './src')
sys.path.append('./src')
from core.common_utils import get_parameters, get_data, get_scaled_train_validation_test_sets, basic_analysis
from core.model_utils import *


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")

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

    # Step 5: Train Model
    VALIDATION_SET = (X_validation, y_validation)
    train_model(model, X_train, y_train, VALIDATION_SET)

    # Step 6: Save Model
    config = get_parameters()
    artifacts_dir = config["ann_mnist_config"]["artifacts_dir"]
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
