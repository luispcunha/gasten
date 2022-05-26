from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_metric_learning import samplers
from dotenv import load_dotenv

from stg.datasets import load_dataset
from stg.metrics.accuracy import binary_accuracy, multiclass_accuracy
from stg.utils import setup_reprod
from stg.utils.checkpoint import checkpoint, construct_classifier_from_checkpoint
from stg.classifier import Classifier


def evaluate(C, device, dataloader, criterion, acc_fun, verbose=True, desc='Validate', header=None):
    training = C.training
    C.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    seq = tqdm(dataloader, desc=desc) if verbose else dataloader

    if header is not None:
        print("\n --- {} ---\n".format(header))

    for i, data in enumerate(seq, 0):
        X, y = data
        X = X.to(device)
        y = y.to(device)

        torch.no_grad()
        y_hat = C(X)
        loss = criterion(y_hat, y)

        running_accuracy += acc_fun(y_hat, y, avg=False)
        running_loss += loss.item() * X.shape[0]

    acc = running_accuracy / len(dataloader.dataset)
    loss = running_loss / len(dataloader.dataset)

    if training:
        C.train()

    return acc.item(), loss


def train(C, opt, crit, train_loader, val_loader, test_loader, acc_fun, args, name, model_params, device):
    stats = {
        'best_loss': float('inf'),
        'best_epoch': 0,
        'early_stop_tracker': 0,
        'cur_epoch': 0,
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    C.train()

    for epoch in range(args.epochs):
        stats['cur_epoch'] = epoch

        print("\n --- Epoch {} ---\n".format(epoch + 1), flush=True)

        ###
        # Train
        ###
        running_accuracy = 0.0
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_loader, desc='Train'), 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()

            y_hat = C(X)
            loss = crit(y_hat, y)
            loss.backward()

            opt.step()

            # Output training stats
            if args.goal_loss_max is not None and args.goal_loss_min is not None:
                print('[%d/%d][%d/%d]\tloss: %.4f\tAcc: %.4f'
                      % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item(),
                         acc_fun(y_hat, y, avg=True)))

                test_acc, test_loss = evaluate(
                    C, device, test_loader, crit, acc_fun, verbose=False)
                stats['test_acc'] = test_acc
                stats['test_loss'] = test_loss

                if args.goal_loss_min <= test_loss <= args.goal_loss_max:
                    name_with_acc = '{}.{}'.format(
                        name, '{}'.format(round(test_loss * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, stats, args,
                                         output_dir=args.out_dir)
                    print("")
                    print(' > Saved checkpoint to {}'.format(cp_path))
                    exit(0)
                elif test_loss < args.goal_loss_min:
                    name_with_acc = '{}.{}_fail'.format(
                        name, '{}'.format(round(test_loss * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, stats, args,
                                         output_dir=args.out_dir)
                    print('')
                    print(' > Saved checkpoint to {}'.format(cp_path))
                    exit(-1)

            running_accuracy += acc_fun(y_hat, y, avg=False)
            running_loss += loss.item() * X.shape[0]

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_accuracy / len(train_loader.dataset)
        stats['train_acc'].append(train_acc.item())
        stats['train_loss'].append(train_loss)

        print("Loss: {}".format(train_loss), flush=True)
        print("Accuracy: {}".format(train_acc), flush=True)

        ###
        # Validation
        ###
        val_acc, val_loss = evaluate(
            C, device, val_loader, crit, acc_fun, verbose=True)
        stats['val_acc'].append(val_acc)
        stats['val_loss'].append(val_loss)

        print("Loss: {}".format(val_loss), flush=True)
        print("Accuracy: {}".format(val_acc), flush=True)

        if val_loss < stats['best_loss']:
            stats['best_loss'] = val_loss
            stats['best_epoch'] = epoch
            stats['early_stop_tracker'] = 0

            cp_path = checkpoint(C, name, model_params,
                                 stats, args, output_dir=args.out_dir)
            print("")
            print(' > Saved checkpoint to {}'.format(cp_path))
        else:
            if args.early_stop is not None:
                stats['early_stop_tracker'] += 1
                print("")
                print(" > Early stop counter: {}/{}".format(
                    stats['early_stop_tracker'], args.early_stop))

                if stats['early_stop_tracker'] == args.early_stop:
                    break

    return stats, cp_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir',
                        default='../data', help='Path to dataset')
    parser.add_argument('--out-dir', dest='out_dir',
                        default='/media/TOSHIBA6T/LCUNHA/msc/classifiers', help='Path to generated files')
    parser.add_argument('--name', dest='name', default='test',
                        help='Name of the classifier for output files')
    parser.add_argument('--dataset', dest='dataset_name',
                        default='mnist', help='Dataset (mnist or fashion-mnist)')
    parser.add_argument('--pos', dest='pos_class', default=7,
                        type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=1,
                        type=int, help='Negative class for binary classification')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--early-stop', dest='early_stop',
                        type=int, default=2, help='Early stopping criteria')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='ADAM opt learning rate')
    parser.add_argument('--goal-loss-min',
                        dest='goal_loss_min', type=float, default=None)
    parser.add_argument('--goal-loss-max',
                        dest='goal_loss_max', type=float, default=None)
    parser.add_argument('--nf', type=int, default=16, help='Num features')
    parser.add_argument('--seed', default=None, type=int, help='Seed')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to run experiments (cpu, cuda:0, cuda:1, ...')

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    print(args)

    seed = np.random.randint(100000) if args.seed is None else args.seed
    setup_reprod(seed)
    args.seed = seed
    print(" > Seed", args.seed)

    device = torch.device("cpu" if args.device is None else args.device)
    print(" > Using device", device)
    name = 'classifier.{}'.format(
        args.dataset_name) if args.name is None else args.name

    dataset, num_classes, img_size = load_dataset(args.dataset_name, args.data_dir,
                                                  pos_class=args.pos_class, neg_class=args.neg_class)
    num_channels = img_size[0]

    print(" > Using dataset", args.dataset_name)
    binary_classification = num_classes == 2

    if binary_classification:
        print("\t> Binary classification between ",
              args.pos_class, "and", args.neg_class)
        name = '{}.{}v{}'.format(
            name, args.pos_class, args.neg_class) if args.name is None else args.name

    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [int(5/6*len(dataset)), len(dataset) - int(5/6*len(dataset))])

    sampler = None
    if args.goal_loss_max is not None and args.goal_loss_min is not None:
        sampler_labels = train_set.dataset.targets[train_set.indices]
        sampler = samplers.MPerClassSampler(
            sampler_labels, args.batch_size / 2, batch_size=args.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True)

    test_set = load_dataset(args.dataset_name, args.data_dir,
                            args.pos_class, args.neg_class, train=False)[0]

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    model_params = {
        'nc': num_channels,
        'nf': args.nf,
        'n_classes': num_classes
    }

    C = Classifier(model_params['nc'], model_params['nf'],
                   model_params['n_classes']).to(device)
    print(C, flush=True)
    opt = optim.Adam(C.parameters(), lr=args.lr)

    if binary_classification:
        criterion = nn.BCELoss()
        acc_fun = binary_accuracy
    else:
        criterion = nn.CrossEntropyLoss()
        acc_fun = multiclass_accuracy

    stats, cp_path = \
        train(C, opt, criterion, train_loader, val_loader,
              test_loader, acc_fun, args, name, model_params, device)

    best_C = construct_classifier_from_checkpoint(cp_path, device=device)[0]
    print("\n")
    print(" > Loading checkpoint from best epoch for testing ...")
    test_acc, test_loss = \
        evaluate(best_C, device, test_loader, criterion,
                 acc_fun, desc='Test', header='Test')

    stats['test_acc'] = test_acc
    stats['test_loss'] = test_loss

    cp_path = checkpoint(best_C, name, model_params, stats,
                         args, output_dir=args.out_dir)
    print('')
    print(' > Saved checkpoint to {}'.format(cp_path))


if __name__ == '__main__':
    main()
