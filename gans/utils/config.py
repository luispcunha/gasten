import os
import yaml
from gans.datasets import valid_dataset
from schema import Schema, SchemaError, Optional, And, Or


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
            Optional("early-stop"): {
                "delta": Or(float, int),
                "criteria": int,
            }
        }),
        "modified-gan": {
            Optional("seed"): int,
            Optional("early-stop"): {
                "delta": Or(float, int),
                "criteria": int,
            },
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

