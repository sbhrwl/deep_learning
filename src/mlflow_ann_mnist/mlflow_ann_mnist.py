import tensorflow as tf
import mlflow.tensorflow
import sys

sys.path.append('./src')
from core.common_utils import get_parameters, get_data, get_scaled_train_validation_test_sets, basic_analysis
from core.model_utils import *
from ann_hyper_parameter_tuning import ann_hyper_parameter_tuning


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
    # basic_analysis(X_train, y_train)

    # Step 4: Create Model
    model = get_model()

    # Step 5: Setup MLFLOW
    config = get_parameters()
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    tensorflow_auto_log_every_n_iter = mlflow_config["tensorflow_auto_log_every_n_iter"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    mlflow.tensorflow.autolog(every_n_iter=tensorflow_auto_log_every_n_iter)

    # Step 6: Train
    VALIDATION_SET = (X_validation, y_validation)
    # train_model(model, X_train, y_train, VALIDATION_SET)
    ann_hyper_parameter_tuning(X_train, y_train)
    mlflow.end_run()

# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts/mlflow-artifacts --host 0.0.0.0 -p 1234
