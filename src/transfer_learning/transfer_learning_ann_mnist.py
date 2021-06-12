import tensorflow as tf
from src.core.common_utils import get_parameters
from src.core.model_utils import get_learning_parameters


def get_model_via_transfer_learning():
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    config = get_parameters()

    model_transfer_learning = config["model_transfer_learning"]
    model_to_load = model_transfer_learning["model_to_load"]
    new_layer_activation = model_transfer_learning["new_layer_activation"]
    new_layer_name = model_transfer_learning["new_layer_name"]

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
    new_model.add(tf.keras.layers.Dense(2, activation=new_layer_activation, name=new_layer_name))
    loss_function, optimizer, metrics = get_learning_parameters()

    new_model.compile(loss=loss_function,
                      optimizer=optimizer,
                      metrics=metrics)
    # print("New_model Summary")
    # print(new_model.summary())
    return new_model
