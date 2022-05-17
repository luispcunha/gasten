import os
from datetime import datetime
import random
import torch
import torch.nn as nn
import numpy as np


def create_checkpoint_path(config):
    path = os.path.join(os.curdir,
                        config['out-dir'],
                        '{}_{}'.format(config['name'], datetime.now().strftime('%m-%dT%H:%M:%S')))

    os.makedirs(path, exist_ok=True)

    return path


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_and_store_z(out_dir, n, dim, name=None):
    if name is None:
        name = "z_{}_{}".format(n, dim)

    noise = torch.randn(n, dim, 1, 1).numpy()
    out_path = os.path.join(out_dir, '{}.npz'.format(name))

    with open(out_path, 'wb') as f:
        np.savez(f, z=noise)

    return torch.Tensor(noise), out_path


def load_z(path):
    with np.load(path) as f:
        z = f['z'][:]

    return torch.Tensor(z)