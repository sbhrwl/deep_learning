import tensorflow as tf
import sys

sys.path.append('./src')
from core.common_utils import get_parameters


def get_model_via_transfer_learning():
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    model_transfer_learning = config["model_transfer_learning"]
    model_to_load = model_transfer_learning["model_to_load"]

    # Step 1: Load Previous model
    model = tf.keras.models.load_model(model_to_load)

    # Step 2: Check Model Details
    # 2.1: Summary
    print(model.summary())

    # 2.2: Trainable layers
    for layer in model.layers:
        print(f"{layer.name}: {layer.trainable}")

    # Step 3: Remove Last Layer
    for layer in model.layers[:-1]:
        layer.trainable = False
        print(f"{layer.name}: {layer.trainable}")

    # Step 4: Create new model
    lower_pretrained_layers = model.layers[:-1]

    new_model = tf.keras.models.Sequential(lower_pretrained_layers)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="NewOutputLayer")
        # tf.keras.layers.Dense(2, activation="binary")
    )

    config = get_parameters()

    model_metrics = config["model_transfer_learning"]["model_metrics"]
    loss_function = model_metrics["loss_function"]
    optimizer = model_metrics["optimizer"]
    metrics = model_metrics["metrics"]

    new_model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                      metrics=metrics)
    # print("New_model Summary")
    # print(new_model.summary())
    return new_model
