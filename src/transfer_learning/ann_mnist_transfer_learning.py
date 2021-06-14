import tensorflow as tf
from src.core.common_utils import get_data, get_scaled_train_validation_test_sets, basic_analysis
from src.core.model_utils import *
from transfer_learning import get_model_via_transfer_learning


if __name__ == "__main__":
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")

    # Step 1: Load data
    (X_train, y_train), (X_test, y_test) = get_data()

    # Step 2: Scale the train, validation and test set
    X_train, y_train, X_validation, y_validation, X_test = get_scaled_train_validation_test_sets(X_train,
                                                                                                 y_train,
                                                                                                 X_test)
    # Step 3: Create Model
    model = get_model_via_transfer_learning("transfer_learning")
    print("New_model Summary")
    print(model.summary())

    # Step 4: Train Model
    VALIDATION_SET = (X_validation, y_validation)
    train_model(model, X_train, y_train, VALIDATION_SET)
