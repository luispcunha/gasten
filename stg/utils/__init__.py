import os
from datetime import datetime
import random
import torch
import torch.nn as nn
import numpy as np
import math
import json
import torchvision.utils as vutils
from .metrics_logger import MetricsLogger


def create_checkpoint_path(config, run_id):
    path = os.path.join(config['out-dir'],
                        config['project'],
                        config['name'],
                        datetime.now().strftime(f'%b%dT%H-%M_{run_id}'))

    os.makedirs(path, exist_ok=True)

    return path


def create_exp_path(config):
    path = os.path.join(config['out-dir'],
                        config['name'])

    os.makedirs(path, exist_ok=True)

    return path


def gen_seed(max_val=10000):
    return np.random.randint(max_val)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_reprod(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_and_store_z(out_dir, n, dim, name=None, config=None):
    if name is None:
        name = "z_{}_{}".format(n, dim)

    noise = torch.randn(n, dim).numpy()
    out_path = os.path.join(out_dir, name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'z.npy'.format(name)), 'wb') as f:
        np.savez(f, z=noise)

    if config is not None:
        with open(os.path.join(out_path, 'z.json'.format(name)), "w") as out_json:
            json.dump(config, out_json)

    return torch.Tensor(noise), out_path


def load_z(path):
    with np.load(os.path.join(path, 'z.npy')) as f:
        z = f['z'][:]

    with open(os.path.join(path, 'z.json')) as f:
        conf = json.load(f)

    return torch.Tensor(z), conf


def make_grid(images):

    nrow = math.sqrt(images.size(0))
    if nrow % 1 != 0:
        nrow = 8

    img = vutils.make_grid(images, padding=2, normalize=True, nrow=int(nrow))

    return img
