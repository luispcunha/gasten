import os
import argparse
import numpy as np
from dotenv import load_dotenv

import torch
import wandb

from stg.metrics import fid, LossSecondTerm
from stg.datasets import load_dataset
from stg.gan.architectures.dcgan import Generator, Discriminator
from stg.gan.train import train
from stg.gan.loss import RegularGeneratorLoss, DiscriminatorLoss, NewGeneratorLossBinary, NewGeneratorLoss
from stg.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, create_exp_path, gen_seed, seed_worker
from stg.utils.config import read_config
from stg.utils.checkpoint import construct_gan_from_checkpoint, construct_classifier_from_checkpoint
from stg.gan import construct_gan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Config file")
    return parser.parse_args()


def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(
        config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(
        config["beta1"], config["beta2"]))

    return g_optim, d_optim


def train_modified_gan(config, dataset, cp_dir, gan_path, test_noise, fid_metrics,
                       C, C_params, C_stats, C_args, classifier_path, weight, fixed_noise, num_classes, device, seed, run_id):
    print("Running experiment with classifier {} and weight {} ...".format(
        classifier_path, weight))

    classifier_name = '.'.join(
        os.path.basename(classifier_path).split('.')[:-1])
    gan_cp_dir = os.path.join(cp_dir, '{}_{}'.format(
        classifier_name, int(weight*100)))

    batch_size = config['train']['step-2']['batch-size']
    n_epochs = config['train']['step-2']['epochs']

    G, D, g_optim, d_optim = construct_gan_from_checkpoint(
        gan_path, device=device)
    d_crit = DiscriminatorLoss()

    if num_classes == 2:
        g_crit = NewGeneratorLossBinary(C, beta=weight)
    else:
        g_crit = NewGeneratorLoss(C, beta=weight)

    early_stop_key = 'conf_dist'
    early_stop_crit = None if 'early-stop' not in config['train']['step-2'] \
        else config['train']['step-2']['early-stop']['criteria']

    set_seed(seed)
    wandb.init(project=config["project"],
               group=config["name"],
               entity="luispcunha",
               job_type='step-2',
               config={
        'id': run_id,
        'seed': seed,
        'train': config["train"]["step-2"],
        'classifier_loss': C_stats['best_loss'],
        'classifier_args': C_args,
        'classifier_params': C_params
    })

    stats, latest_checkpoint_dir = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_crit,
        D, d_optim, d_crit,
        test_noise, fid_metrics,
        early_stop_key=early_stop_key, early_stop_crit=early_stop_crit,
        checkpoint_dir=gan_cp_dir, fixed_noise=fixed_noise)


def compute_dataset_fid_stats(dset, get_feature_map_fn, dims, batch_size=64, device='cpu', num_workers=0):
    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    m, s = fid.calculate_activation_statistics_dataloader(
        dataloader, get_feature_map_fn, dims=dims, device=device)

    return m, s


def main():
    load_dotenv()
    args = parse_args()

    config = read_config(args.config_path)
    print("Loaded experiment configuration from {}".format(args.config_path))

    if "step-1-seeds" not in config:
        run_seeds = [gen_seed() for _ in range(config["num-runs"])]
    else:
        run_seeds = config["step-1-seeds"]

    if "step-2-seeds" not in config:
        step_2_seeds = [gen_seed() for _ in range(config["num-runs"])]
    else:
        step_2_seeds = config["step-2-seeds"]

    device = torch.device(config["device"])
    print("Using device {}".format(device))

    ###
    # Setup
    ###
    pos_class = None
    neg_class = None
    if "binary" in config["dataset"]:
        pos_class = config["dataset"]["binary"]["pos"]
        neg_class = config["dataset"]["binary"]["neg"]

    dataset, num_classes, img_size = load_dataset(
        config["dataset"]["name"], config["data-dir"], pos_class, neg_class)
    g_crit = RegularGeneratorLoss()
    d_crit = DiscriminatorLoss()

    num_workers = config["num-workers"]
    print(" > Num workers", num_workers)

    if type(config['fixed-noise']) == str:
        arr = np.load(config['fixed-noise'])
        fixed_noise = torch.Tensor(arr).to(device)
    else:
        fixed_noise = torch.randn(
            config['fixed-noise'], config["model"]["z_dim"], device=device)

    test_noise, test_noise_conf = load_z(config['test-noise'])
    print("Loaded test noise from", config['test-noise'])
    print("\t", test_noise_conf)

    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    num_runs = config["num-runs"]
    for i in range(num_runs):
        print("##")
        print("# Starting run", i)
        print("##")

        run_id = wandb.util.generate_id()
        cp_dir = create_checkpoint_path(config, run_id)
        with open(os.path.join(cp_dir, 'fixed_noise.npy'), 'wb') as f:
            np.save(f, fixed_noise.cpu().numpy())

        #################
        # Set seed
        seed = run_seeds[i]
        setup_reprod(seed)

        config["model"]["image-size"] = img_size

        G, D = construct_gan(
            config["model"], img_size, device)
        g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

        print("Storing generated artifacts in {}".format(cp_dir))
        original_gan_cp_dir = os.path.join(cp_dir, 'step-1')

        ###
        # Step 1 (train GAN with normal GAN loss)
        ###
        if type(config['train']['step-1']) != str:
            batch_size = config['train']['step-1']['batch-size']
            n_epochs = config['train']['step-1']['epochs']

            fid_metrics = {
                'fid': original_fid
            }
            early_stop_key = 'fid'
            early_stop_crit = None if 'early-stop' not in config['train']['step-1'] \
                else config['train']['step-1']['early-stop']['criteria']

            wandb.init(project=config["project"],
                       group=config["name"],
                       entity="luispcunha",
                       job_type='step-1',
                       config={
                           'id': run_id,
                           'seed': seed,
                           'gan': config["model"],
                           'optim': config["optimizer"],
                           'train': config["train"]["step-1"],
                           'dataset': config["dataset"],
                           'num-workers': config["num-workers"],
                           'test-noise': test_noise_conf,
            })

            _, best_cp_dir = train(
                config, dataset, device, n_epochs, batch_size,
                G, g_optim, g_crit,
                D, d_optim, d_crit,
                test_noise, fid_metrics,
                early_stop_crit=early_stop_crit, early_stop_key=early_stop_key,
                checkpoint_dir=original_gan_cp_dir, fixed_noise=fixed_noise)

            wandb.finish()
        else:
            best_cp_dir = config['train']['step-1']

        print(" > Start step 2 (gan with modified loss")
        ###
        # Train modified GAN
        ###
        classifier_paths = config['train']['step-2']['classifier']
        weights = config['train']['step-2']['weight']

        mod_gan_seed = step_2_seeds[i]

        for c_path in classifier_paths:
            C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(
                c_path, device=device)
            C.to(device)
            C.eval()
            C.output_feature_maps = True

            def get_feature_map_fn(batch):
                return C(batch, output_feature_maps=True)[-2]

            dims = get_feature_map_fn(dataset.data[0:1].to(device)).size(1)

            print(" > Computing statistics using original dataset")
            mu, sigma = compute_dataset_fid_stats(
                dataset, get_feature_map_fn, dims, device=device, num_workers=num_workers)
            print("   ... done")

            our_class_fid = fid.FID(get_feature_map_fn, dims,
                                    test_noise.size(0), mu, sigma, device=device)
            conf_dist = LossSecondTerm(C)

            fid_metrics = {
                'fid': original_fid,
                'focd': our_class_fid,
                'conf_dist': conf_dist,
            }

            for weight in weights:
                train_modified_gan(config, dataset, cp_dir, best_cp_dir,
                                   test_noise, fid_metrics,
                                   C, C_params, C_stats, C_args, c_path,
                                   weight, fixed_noise, num_classes, device, mod_gan_seed, run_id)


if __name__ == "__main__":
    main()
