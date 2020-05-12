import yaml


def read_yaml(path: str):
    with open(path, "r") as fp:
        res = yaml.load(fp)
    return res
