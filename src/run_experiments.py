import os
import argparse
import numpy as np

import torch

from metrics import fid, LossSecondTerm
from datasets import load_dataset
from gan.architectures.dcgan import Generator, Discriminator
from gan.train import train
from gan.loss import RegularGeneratorLoss, DiscriminatorLoss, NewGeneratorLossBinary, NewGeneratorLoss
from utils import weights_init, create_and_store_z, load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed
from utils.config import read_config
from utils.checkpoint import construct_gan_from_checkpoint, construct_classifier_from_checkpoint
from utils.plot import plot_train_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Config file")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", default=True)

    return parser.parse_args()


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


def train_modified_gan(config, dataset, cp_dir, gan_path, test_noise, fid_metrics,
                       C, classifier_path, weight, fixed_noise, num_classes, device, seed):
    print("Running experiment with classifier {} and weight {} ...".format(classifier_path, weight))

    classifier_name = '.'.join(os.path.basename(classifier_path).split('.')[:-1])
    gan_cp_dir = os.path.join(cp_dir, '{}_{}'.format(classifier_name, weight))

    batch_size = config['train']['modified-gan']['batch-size']
    n_epochs = config['train']['modified-gan']['epochs']

    G, D, g_optim, d_optim = construct_gan_from_checkpoint(gan_path, device=device)
    d_crit = DiscriminatorLoss()

    if num_classes == 2:
        g_crit = NewGeneratorLossBinary(C, beta=weight)
    else:
        g_crit = NewGeneratorLoss(C, beta=weight)

    early_stop_key = 'conf_dist'
    early_stop_crit = None if 'early-stop' not in config['train']['original-gan'] \
        else config['train']['original-gan']['early-stop']['criteria']

    set_seed(seed)
    stats, images, latest_checkpoint_dir = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_crit,
        D, d_optim, d_crit,
        test_noise, fid_metrics,
        early_stop_key=early_stop_key, early_stop_crit=early_stop_crit,
        checkpoint_dir=gan_cp_dir, fixed_noise=fixed_noise)

    plot_train_summary(stats, os.path.join(cp_dir, 'plots'))


def compute_dataset_fid_stats(dset, get_feature_map_fn, dims, batch_size=64, device='cpu'):
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

    m, s = fid.calculate_activation_statistics_dataloader(dataloader, get_feature_map_fn, dims=dims, device=device)

    return m, s


def main():
    args = parse_args()

    config = read_config(args.config_path)
    print("Loaded experiment configuration from {}".format(args.config_path))

    if "seed" not in config:
        config["seed"] = gen_seed()
    if "seed" not in config["train"]["modified-gan"]:
        config["train"]["modified-gan"]["seed"] = gen_seed()

    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("Using device {}".format(device))

    ###
    # Setup
    ###
    pos_class = None
    neg_class = None
    if "binary" in config["dataset"]:
        pos_class = config["dataset"]["binary"]["pos"]
        neg_class = config["dataset"]["binary"]["neg"]

    dataset, num_classes, _, _ = load_dataset(config["dataset"]["name"], config["dataset"]["dir"], pos_class, neg_class)
    g_crit = RegularGeneratorLoss()
    d_crit = DiscriminatorLoss()

    # Set seed

    seed = config["seed"]
    setup_reprod(seed)

    G, D = construct_gan(config["model"], device)
    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

    cp_dir = create_checkpoint_path(config)
    print("Storing generated artifacts in {}".format(cp_dir))
    original_gan_cp_dir = os.path.join(cp_dir, 'original')

    if type(config['test-noise']) == str:
        test_noise = load_z(config['test-noise'])
        print("Loaded test noise from", config['test-noise'])
    else:
        test_noise, test_noise_path = create_and_store_z(
            cp_dir, config['test-noise']['n'], config['model']['nz'])
        print("Generated test noise, stored in", test_noise_path)

    if type(config['fixed-noise']) == str:
        arr = np.load(config['fixed-noise'])
        fixed_noise = torch.Tensor(arr).to(device)
    else:
        fixed_noise = torch.randn(config['fixed-noise'], G.nz, 1, 1, device=device)
        with open(os.path.join(cp_dir, 'fixed_noise.npy'), 'wb') as f:
            np.save(f, fixed_noise.cpu().numpy())

    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    ###
    # Train original GAN
    ###
    if type(config['train']['original-gan']) != str:
        batch_size = config['train']['original-gan']['batch-size']
        n_epochs = config['train']['original-gan']['epochs']

        fid_metrics = {
            'fid': original_fid
        }
        early_stop_key = 'fid'
        early_stop_crit = None if 'early-stop' not in config['train']['original-gan'] \
            else config['train']['original-gan']['early-stop']['criteria']

        stats, images, best_cp_dir = train(
            config, dataset, device, n_epochs, batch_size,
            G, g_optim, g_crit,
            D, d_optim, d_crit,
            test_noise, fid_metrics,
            early_stop_crit=early_stop_crit, early_stop_key=early_stop_key,
            checkpoint_dir=original_gan_cp_dir, fixed_noise=fixed_noise)

        plot_train_summary(stats, os.path.join(original_gan_cp_dir, 'plots'))
    else:
        best_cp_dir = config['train']['original-gan']

    print("Start train mod gan")

    ###
    # Train modified GAN
    ###
    classifier_paths = config['train']['modified-gan']['classifier']
    weights = config['train']['modified-gan']['weight']

    mod_gan_seed = config['train']['modified-gan']['seed']

    for c_path in classifier_paths:
        C = construct_classifier_from_checkpoint(c_path, device=device)[0]
        C.to(device)
        C.eval()
        C.output_feature_maps = True

        def get_feature_map_fn(batch):
            return C(batch, output_feature_maps=True)[-2]

        dims = get_feature_map_fn(dataset.data[0:1].to(device)).size(1)

        print(" > Computing statistics using original dataset")
        mu, sigma = compute_dataset_fid_stats(dataset, get_feature_map_fn, dims, device=device)
        print("   ... done")

        our_class_fid = fid.FID(get_feature_map_fn, dims, test_noise.size(0), mu, sigma, device=device)
        conf_dist = LossSecondTerm(C)

        fid_metrics = {
            'fid': original_fid,
            'focd': our_class_fid,
            'conf_dist': conf_dist,
        }

        for weight in weights:
            train_modified_gan(config, dataset, cp_dir, best_cp_dir, test_noise, fid_metrics,
                               C, c_path, weight, fixed_noise, num_classes, device, mod_gan_seed)


if __name__ == "__main__":
    main()
