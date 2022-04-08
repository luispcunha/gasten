import os
import argparse
from datetime import datetime
import itertools
import numpy as np

import torch

from config import read_config
from datasets import get_mnist
from binary_dataset import BinaryDataset
from discriminator import Discriminator
from generator import Generator
from utils import weights_init
from train_gan import train
from loss import RegularGeneratorLoss, DiscriminatorLoss, NewGeneratorLossBinary
from checkpoint_utils import construct_gan_from_checkpoint, construct_classifier_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Config file")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", default=True)

    return parser.parse_args()


def load_dataset(config):
    if config["name"].lower() != "mnist":
        print("{} dataset not supported".format(config["name"]))
        exit(-1)

    dataset = get_mnist(config["dir"])
    num_classes = dataset.targets.unique().size()

    if "binary" in config:
        num_classes = 2
        pos_class = config["binary"]["pos"]
        neg_class = config["binary"]["neg"]

        dataset = BinaryDataset(dataset, pos_class, neg_class)
        print("Loaded {} dataset for binary classification ({} vs. {})".format(config["name"], pos_class, neg_class))

    return dataset, num_classes


def construct_gan(config, device):
    G = Generator(config['nc'], ngf=config['ngf'], nz=config['nz']).to(device)
    D = Discriminator(config['nc'], ndf=config['ndf']).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    return G, D


def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))

    return g_optim, d_optim


def create_checkpoint_dir(config):
    path = os.path.join(os.curdir,
                        config['out-dir'],
                        '{}_{}'.format(config['name'], datetime.now().strftime('%m-%dT%H:%M:%S')))

    os.makedirs(path, exist_ok=True)

    return path


def train_modified_gan(config, dataset, cp_dir, gan_path, classifier_path, weight, fixed_noise, device):
    print("Running experiment with classifier {} and weight {} ...".format(classifier_path, weight))

    classifier_name = '.'.join(os.path.basename(classifier_path).split('.')[:-1])
    gan_cp_dir = os.path.join(cp_dir, '{}_{}'.format(classifier_name, weight))

    batch_size = config['train']['modified-gan']['batch-size']
    n_epochs = config['train']['modified-gan']['epochs']

    C = construct_classifier_from_checkpoint(classifier_path, device=device)[0]
    G, D, g_optim, d_optim = construct_gan_from_checkpoint(gan_path, device=device)
    d_crit = DiscriminatorLoss()
    g_crit = NewGeneratorLossBinary(C, beta=weight)

    stats, images, latest_checkpoint_dir = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_crit,
        D, d_optim, d_crit,
        gan_cp_dir, fixed_noise=fixed_noise)


def main():
    args = parse_args()

    config = read_config(args.config_path)
    print("Loaded experiment configuration from {}".format(args.config_path))

    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("Using device {}".format(device))

    ###
    # Setup
    ###
    dataset, num_classes = load_dataset(config["dataset"])
    G, D = construct_gan(config["model"], device)
    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)
    g_crit = RegularGeneratorLoss()
    d_crit = DiscriminatorLoss()

    cp_dir = create_checkpoint_dir(config)
    print("Storing generated artifacts in {}".format(cp_dir))

    original_gan_cp_dir = os.path.join(cp_dir, 'original')

    fixed_noise = None
    if type(config['fixed-noise']) == str:
        arr = np.load(config['fixed-noise'])
        fixed_noise = torch.Tensor(arr).to(device)
    else:
        fixed_noise = torch.randn(config['fixed-noise-size'], G.nz, 1, 1, device=device)
        with open(os.path.join(cp_dir, 'fixed_noise.npy'), 'wb') as f:
            np.save(f, fixed_noise.cpu().numpy())

    ###
    # Train original GAN
    ###
    if type(config['train']['original-gan']) != str:
        batch_size = config['train']['original-gan']['batch-size']
        n_epochs = config['train']['original-gan']['epochs']

        stats, images, latest_checkpoint_dir = train(
            config, dataset, device, n_epochs, batch_size,
            G, g_optim, g_crit,
            D, d_optim, d_crit,
            original_gan_cp_dir, fixed_noise=fixed_noise)
    else:
        latest_checkpoint_dir = config['train']['original-gan']

    ###
    # Train modified GAN
    ###
    classifier_paths = config['train']['modified-gan']['classifier']
    weights = config['train']['modified-gan']['weight']

    for c_path, weight in itertools.product(classifier_paths, weights):
        train_modified_gan(config, dataset, cp_dir, latest_checkpoint_dir, c_path, weight, fixed_noise, device)


if __name__ == "__main__":
    main()
