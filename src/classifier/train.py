from tqdm import tqdm
from src.utils.checkpoint import load_checkpoint, checkpoint


def evaluate(C, device, dataloader, criterion, acc_fun, verbose=True, desc='Validate', header=None):
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

        y_hat = C(X)
        loss = criterion(y_hat, y)

        running_accuracy += acc_fun(y_hat, y, avg=False)
        running_loss += loss.item() * X.shape[0]

    acc = running_accuracy / len(dataloader.dataset)
    loss = running_loss / len(dataloader.dataset)

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

    for epoch in range(args.epochs):
        stats['cur_epoch'] = epoch

        print("\n --- Epoch {} ---\n".format(epoch + 1), flush=True)

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
            loss = crit(y_hat, y)
            loss.backward()

            opt.step()

            # Output training stats
            if args.goal_loss_max is not None and args.goal_loss_min is not None:
                print('[%d/%d][%d/%d]\tloss: %.4f\tAcc: %.4f'
                      % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item(),
                         acc_fun(y_hat, y, avg=True)))

                test_acc, test_loss = evaluate(C, device, test_loader, crit, acc_fun, verbose=False)
                stats['test_acc'] = test_acc
                stats['test_loss'] = test_loss

                if args.goal_loss_min <= test_loss <= args.goal_loss_max:
                    name_with_acc = '{}.{}'.format(name, '{}'.format(round(test_loss * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, stats, args,
                                         output_dir=args.out_dir)
                    print("")
                    print(' > Saved checkpoint to {}'.format(cp_path))
                    exit(0)
                elif test_loss < args.goal_loss_min:
                    name_with_acc = '{}.{}_fail'.format(name, '{}'.format(round(test_loss * 100)))
                    cp_path = checkpoint(C, name_with_acc, model_params, stats, args,
                                         output_dir=args.out_dir)
                    print('')
                    print(' > Saved checkpoint to {}'.format(cp_path))
                    exit(-1)

                C.train()

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
        val_acc, val_loss = evaluate(C, device, val_loader, crit, acc_fun, verbose=True)
        stats['val_acc'].append(val_acc)
        stats['val_loss'].append(val_loss)

        print("Loss: {}".format(val_loss), flush=True)
        print("Accuracy: {}".format(val_acc), flush=True)

        if val_loss < stats['best_loss']:
            stats['best_loss'] = val_loss
            stats['best_epoch'] = epoch
            stats['early_stop_tracker'] = 0

            cp_path = checkpoint(C, name, model_params, stats, args, output_dir=args.out_dir)
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
