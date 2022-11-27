from pathlib import Path


def curdir(path: str):
    return Path(path).absolute().parent
