import yaml
import argparse


def get_parameters():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="parameters.yaml")
    parsed_args = args.parse_args()

    with open(parsed_args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    data = get_parameters()
    print('Parameters: ', data)
