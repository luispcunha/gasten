import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import samplers

from datasets import get_mnist, get_fashion_mnist
from utils import binary_accuracy, multiclass_accuracy
from binary_dataset import BinaryDataset
from checkpoint_utils import checkpoint, load_checkpoint
from classifier import Classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', default='../data', help='Path to dataset')
    parser.add_argument('--out-dir', dest='out_dir', default='../out', help='Path to generated files')
    parser.add_argument('--name', dest='name', default=None, help='Name of the classifier for output files')
    parser.add_argument('--dataset', dest='dataset_name', default='fashion-mnist', help='Dataset (mnist or fashion-mnist)')
    parser.add_argument('--pos', dest='pos_class', default=None, type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=None, type=int, help='Negative class for binary classification')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--image-size', dest='image_size', type=int, default=28, help='Height / width of the images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--early-stop', type=int, default=2, help='Early stopping criteria')
    parser.add_argument('--lr', type=float, default=1e-3, help='ADAM opt learning rate')
    parser.add_argument('--goal-loss-min', dest='goal_loss_min', type=float, default=None)
    parser.add_argument('--goal-loss-max', dest='goal_loss_max', type=float, default=None)
    parser.add_argument('--nc', type=int, default=1, help='Num channels')
    parser.add_argument('--nf', type=int, default=32, help='Num features')

    return parser.parse_args()


def test(C, device, test_set, test_loader, criterion, acc_fun, verbose=True, cp_path=None):
    ###
    # Test
    ###
    if cp_path is not None:
        print("\nLoading checkpoint from best epoch for testing ...")
        load_checkpoint(cp_path, C, device=device)

    C.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    seq = tqdm(test_loader, desc='Test') if verbose else test_loader

    print("\n --- Test ---\n")
    for i, data in enumerate(seq, 0):
        X, y = data
        X = X.to(device)
        y = y.to(device)

        y_hat = C(X)
        loss = criterion(y_hat, y)

        running_accuracy += acc_fun(y_hat, y, avg=False)
        running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_accuracy / len(test_set)

    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_accuracy))

    return {'acc': test_accuracy.item(), 'loss': test_loss}


def main():
    args = parse_args()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    name = 'classifier.{}'.format(args.dataset_name) if args.name is None else args.name

    if args.dataset_name == 'fashion-mnist':
        get_dataset = get_fashion_mnist
    elif args.dataset_name == 'mnist':
        get_dataset = get_mnist
    else:
        print('Dataset {} not supported'.format(args.dataset_name))
        exit(-1)

    dataset = get_dataset(args.data_dir, args.image_size, train=True)
    num_classes = dataset.targets.unique().size()

    binary_classification = args.pos_class is not None and args.neg_class is not None

    if binary_classification:
        name = '{}.{}v{}'.format(name, args.pos_class, args.neg_class) if args.name is None else args.name
        num_classes = 2
        dataset = BinaryDataset(dataset, args.pos_class, args.neg_class)

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

    test_set = get_dataset(args.data_dir, args.image_size, train=False)
    if binary_classification:
        test_set = BinaryDataset(test_set, args.pos_class, args.neg_class)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    model_params = {
        'nc': args.nc,
        'nf': args.nf,
        'n_classes': num_classes
    }

    C = Classifier(model_params['nc'], model_params['nf'], model_params['n_classes']).to(device)
    print(C)
    opt = optim.Adam(C.parameters(), lr=args.lr)

    if binary_classification:
        criterion = nn.BCELoss()
        acc_fun = binary_accuracy
    else:
        criterion = nn.CrossEntropyLoss()
        acc_fun = multiclass_accuracy

    best_loss = float('inf')
    best_epoch = 0
    early_stop_tracker = 0
    early_stop = False

    for epoch in range(args.epochs):
        if early_stop:
            break
        print("\n --- Epoch {} ---\n".format(epoch+1), flush=True)

        ###
        # Train
        ###
        C.train()
        running_accuracy = 0.0
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_loader, desc='Train'), 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()

            y_hat = C(X)
            loss = criterion(y_hat, y)
            loss.backward()

            opt.step()

            # Output training stats
            if args.goal_loss_max is not None and args.goal_loss_min is not None:
                print('[%d/%d][%d/%d]\tloss: %.4f\tAcc: %.4f'
                      % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item(),
                         binary_accuracy(y_hat, y, avg=True)))
                test_stats = test(C, device, test_set, test_loader, criterion, acc_fun, cp_path=None, verbose=False)
                if args.goal_loss_min <= test_stats['loss'] <= args.goal_loss_max:
                    name_with_acc = '{}.{}'.format(name, '{}'.format(round(test_stats['loss'] * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, {'test': test_stats}, args,
                                         output_dir=args.out_dir)
                    print("\nSaved checkpoint to {}".format(cp_path))
                    exit(0)
                elif test_stats['loss'] < args.goal_loss_min:
                    name_with_acc = '{}.{}_fail'.format(name, '{}'.format(round(test_stats['loss'] * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, {'test': test_stats}, args,
                                         output_dir=args.out_dir)
                    print("\nSaved checkpoint to {}".format(cp_path))
                    exit(-1)

                C.train()

            running_accuracy += acc_fun(y_hat, y, avg=False)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_set)

        print("Loss: {}".format(epoch_loss), flush=True)
        print("Accuracy: {}".format(epoch_accuracy), flush=True)

        ###
        # Validation
        ###
        C.eval()

        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(tqdm(val_loader, desc='Validation'), 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            y_hat = C(X)
            loss = criterion(y_hat, y)

            running_accuracy += acc_fun(y_hat, y, avg=False)
            running_loss += loss.item()

        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = running_accuracy / len(val_set)

        print("Loss: {}".format(epoch_loss), flush=True)
        print("Accuracy: {}".format(epoch_accuracy), flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            early_stop_tracker = 0

            cp_path = checkpoint(C, name, model_params, '', args, output_dir=args.out_dir)
            print("\nSaved checkpoint to {}".format(cp_path))
        else:
            if args.early_stop is not None:
                early_stop_tracker += 1
                print("\nEarly stop counter: {}/{}".format(
                    early_stop_tracker, args.early_stop))
                if early_stop_tracker == args.early_stop:
                    early_stop = True

    test_stats = test(C, device, test_set, test_loader, criterion, acc_fun, cp_path=cp_path)
    cp_path = checkpoint(C, name, model_params, {'test': test_stats}, args,
                         output_dir=args.out_dir)


if __name__ == '__main__':
    main()
