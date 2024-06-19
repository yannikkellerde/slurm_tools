import yaml


def load_yaml(path):
    if path is None or len(path) == 0:
        return {}
    with open(path, "r") as file:
        return yaml.safe_load(file)
