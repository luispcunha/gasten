import os
import yaml
import numpy as np
from schema import Schema, SchemaError, Optional, And, Or


def valid_dataset(name):
    valid_ds = {"mnist", "fashion-mnist"}
    return name.lower() in valid_ds


def gen_seed():
    return np.random.randint(100000)


config_schema = Schema({
    "name": str,
    "out-dir": os.path.exists,
    "fid-stats-path": os.path.exists,
    "fixed-noise": Or(str, int),
    "test-noise": Or(And(str, os.path.exists), {
        "n": int
    }),
    Optional("seed", default=10): int,
    "dataset": {
        "dir": str,
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "model": {
        "nc": int,
        "nz": int,
        "ngf": int,
        "ndf": int,
    },
    "optimizer": {
        "lr": float,
        "beta1": float,
        "beta2": float,
    },
    "train": {
        "original-gan": Or(And(str, os.path.exists), {
            "epochs": int,
            "batch-size": int,
        }),
        "modified-gan": {
            Optional("seed", default=gen_seed): int,
            "epochs": int,
            "batch-size": int,
            "classifier": [And(str, os.path.exists)],
            "weight": [And(Or(int, float), lambda n: 0 <= n <= 1)]
        }

    }
})


def read_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        config_schema.validate(config)
    except SchemaError as se:
        raise se

    return config

