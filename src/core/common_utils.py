import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def get_parameters():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="parameters.yaml")
    parsed_args = args.parse_args()

    with open(parsed_args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data():
    mnist = tf.keras.datasets.mnist
    (features_train, target_train), (features_test, target_test) = mnist.load_data()
    print(f"data type of features_train: {features_train.dtype},\nshape of features_train: {features_train.shape}")
    return (features_train, target_train), (features_test, target_test)


def get_scaled_train_validation_test_sets(features_train, target_train, features_test):
    features_validation = features_train[:5000] / 255.
    features_train = features_train[5000:] / 255.
    target_validation = target_train[:5000]
    target_train = target_train[5000:]
    features_test = features_test / 255.
    return features_train, target_train, features_validation, target_validation, features_test


def basic_analysis(features_train, target_train):
    plt.imshow(features_train[0], cmap="binary")
    plt.axis('off')
    # plt.show()

    plt.figure(figsize=(15, 15))
    sns.heatmap(features_train[0], annot=True, cmap="binary")
    # actual value of target_train
    # print(target_train[0])
