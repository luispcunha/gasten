import os
import torch
import torch.nn as nn
import numpy as np


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