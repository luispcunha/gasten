import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from metrics import fid
from metrics.fid import get_inception_feature_map_fn
from utils.checkpoint import construct_classifier_from_checkpoint
from datasets import get_mnist
from datasets.utils import BinaryDataset


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='dataroot', default='../data', help='Dir with dataset')
parser.add_argument('--dataset', dest='dataset', default='mnist', help='Dataset (mnist or fashion-mnist)')
parser.add_argument('--pos', dest='pos_class', default=7, type=int, help='Positive class for binary classification')
parser.add_argument('--neg', dest='neg_class', default=1, type=int, help='Negative class for binary classification')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--model-path', dest='model_path', default='../out/test_fid.pth', type=str,
                    help=('Path to classifier to use'
                          'If none, uses InceptionV3'))
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--name', dest='name', default=None, help='name of gen .npz file')


def main():
    args = parser.parse_args()
    print(args)

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)
    print("Using device", device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    print("Num workers", num_workers)

    if args.dataset == 'mnist':
        dset = get_mnist(args.dataroot)
    else:
        print("invalid dataset", args.dataset)
        exit(-1)

    name = '{}_stats'.format(args.dataset) if args.name is None else args.name

    binary_classification = args.pos_class is not None and args.neg_class is not None
    if binary_classification:
        name = '{}.{}v{}'.format(name, args.pos_class, args.neg_class)
        dset = BinaryDataset(dset, args.pos_class, args.neg_class)

    print('Dataset size', len(dset))

    dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    if args.model_path is None:
        get_feature_map_fn, dims = get_inception_feature_map_fn(device)
    else:
        if not os.path.exists(args.model_path):
            print("Model Path doesn't exist")
            exit(-1)
        model = construct_classifier_from_checkpoint(args.model_path)[0]
        model.to(device)
        model.eval()
        model.output_feature_maps = True

        def get_feature_map_fn(batch):
            return model(batch)[-2]

        dims = get_feature_map_fn(dset.data[0:1]).size(1)

    m, s = fid.calculate_activation_statistics_dataloader(dataloader, get_feature_map_fn, dims=dims, device=device)
    with open(os.path.join(args.dataroot, '{}.npz'.format(name)), 'wb') as f:
        np.savez(f, mu=m, sigma=s)


if __name__ == '__main__':
    main()
