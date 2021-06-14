import numpy as np
from src.core.common_utils import get_data, get_scaled_train_validation_test_sets
from src.core.model_utils import *
from transfer_learning import get_model_via_transfer_learning


def update_even_odd_labels(labels):
    for idx, label in enumerate(labels):
        labels[idx] = np.where(label % 2 == 0, 1, 0)
    return labels


if __name__ == "__main__":
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
    model = get_model_via_transfer_learning("yes")
    print("New_model Summary")
    print(model.summary())

    # Step 5: Train
    VALIDATION_SET = (X_validation, y_validation_binary)
    train_model(model, X_train, y_train_binary, VALIDATION_SET)

    # Step 6: Evaluate
    model.evaluate(X_test, y_test_binary)

    # 6.1: Evaluation Test
    X_new = X_test[:3]
    y_test[:3], y_test_binary[:3]

    # 6.2: Verify Predictions
    predictions = np.argmax(model.predict(X_new), axis=-1)
    # model.predict(X_new)
    print(predictions)
