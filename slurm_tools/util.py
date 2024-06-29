def load_yaml(path):
    import yaml

    if path is None or len(path) == 0:
        return {}
    with open(path, "r") as file:
        return yaml.safe_load(file)
