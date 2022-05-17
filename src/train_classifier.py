import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_metric_learning import samplers

from datasets import load_dataset
from metrics.accuracy import binary_accuracy, multiclass_accuracy
from src.utils import setup_reprod
from utils.checkpoint import checkpoint, construct_classifier_from_checkpoint
from classifier import Classifier
from classifier.train import train, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', default='../data', help='Path to dataset')
    parser.add_argument('--out-dir', dest='out_dir', default='../out', help='Path to generated files')
    parser.add_argument('--name', dest='name', default=None, help='Name of the classifier for output files')
    parser.add_argument('--dataset', dest='dataset_name', default='mnist', help='Dataset (mnist or fashion-mnist)')
    parser.add_argument('--pos', dest='pos_class', default=7, type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=1, type=int, help='Negative class for binary classification')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--early-stop', dest='early_stop', type=int, default=2, help='Early stopping criteria')
    parser.add_argument('--lr', type=float, default=1e-3, help='ADAM opt learning rate')
    parser.add_argument('--goal-loss-min', dest='goal_loss_min', type=float, default=None)
    parser.add_argument('--goal-loss-max', dest='goal_loss_max', type=float, default=None)
    parser.add_argument('--nf', type=int, default=16, help='Num features')
    parser.add_argument('--seed', default=None, type=int, help='Seed')

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    seed = np.random.randint(100000) if args.seed is None else args.seed
    setup_reprod(seed)
    args.seed = seed
    print(" > Seed", args.seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(" > Using device", device)

    name = 'classifier.{}'.format(args.dataset_name) if args.name is None else args.name

    dataset, num_classes, num_channels, img_size = load_dataset(args.dataset_name, args.data_dir,
                                                                pos_class=args.pos_class, neg_class=args.neg_class)

    print(" > Using dataset", dataset)
    binary_classification = num_classes == 2

    if binary_classification:
        print("\t> Binary classification between ", args.pos_class, "and", args.neg_class)
        name = '{}.{}v{}'.format(name, args.pos_class, args.neg_class) if args.name is None else args.name

    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [int(5/6*len(dataset)), len(dataset) - int(5/6*len(dataset))])

    sampler = None
    if args.goal_loss_max is not None and args.goal_loss_min is not None:
        sampler_labels = train_set.dataset.targets[train_set.indices]
        sampler = samplers.MPerClassSampler(sampler_labels, args.batch_size / 2, batch_size=args.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True)

    test_set = load_dataset(args.dataset_name, args.data_dir, args.pos_class, args.neg_class, train=False)[0]

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    model_params = {
        'nc': num_channels,
        'nf': args.nf,
        'n_classes': num_classes
    }

    C = Classifier(model_params['nc'], model_params['nf'], model_params['n_classes']).to(device)
    print(C, flush=True)
    opt = optim.Adam(C.parameters(), lr=args.lr)

    if binary_classification:
        criterion = nn.BCELoss()
        acc_fun = binary_accuracy
    else:
        criterion = nn.CrossEntropyLoss()
        acc_fun = multiclass_accuracy

    stats, cp_path = \
        train(C, opt, criterion, train_loader, val_loader, test_loader, acc_fun, args, name, model_params, device)

    best_C = construct_classifier_from_checkpoint(cp_path, device=device)[0]
    print("\n")
    print(" > Loading checkpoint from best epoch for testing ...")
    test_acc, test_loss = \
        evaluate(best_C, device, test_loader, criterion, acc_fun, desc='Test', header='Test')

    stats['test_acc'] = test_acc
    stats['test_loss'] = test_loss

    cp_path = checkpoint(best_C, name, model_params, stats, args, output_dir=args.out_dir)
    print('')
    print(' > Saved checkpoint to {}'.format(cp_path))


if __name__ == '__main__':
    main()
