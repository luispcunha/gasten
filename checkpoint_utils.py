import os
import json
import torch
import torchvision.utils as vutils
import torch.optim as optim
from classifier import Classifier


def checkpoint(model, model_name, model_params, train_stats, args, output_dir=None, optimizer=None):
    output_dir = os.path.curdir if output_dir is None else output_dir

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, '{}.pth'.format(model_name))

    save_dict = {
        'name': model_name,
        'state': model.state_dict(),
        'stats': train_stats,
        'params': model_params,
        'args': args
    }

    json.dump({'train_stats': train_stats, 'model_params': model_params, 'args': vars(args)},
              open(os.path.join(output_dir, '{}.json'.format(model_name)), 'w'), indent=2)

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    torch.save(save_dict, path)

    return path


def load_checkpoint(path, model, device=None, optimizer=None):
    cp = torch.load(path, map_location=device)

    model.load_state_dict(cp['state'])
    model.eval()  # just to be safe

    if optimizer is not None:
        optimizer.load_state_dict(cp['optimizer'])


def construct_classifier_from_checkpoint(path, device=None, optimizer=False):
    cp = torch.load(path, map_location=device)

    print("Loading model from {} ...".format(path))

    model_params = cp['params']
    print('Model', cp['name'])
    print('\t> Params: ', model_params)

    model = Classifier(model_params['nc'], model_params['nf']).to(device)
    model.load_state_dict(cp['state'])
    model.eval()

    if optimizer is True:
        opt = optim.Adam(model.parameters())
        opt.load_state_dict(cp['optimizer'].state_dict())
        return model, model_params, cp['stats'], cp['args'], opt
    else:
        return model, model_params, cp['stats'], cp['args']


def checkpoint_gan(G, D, g_opt, d_opt, stats, output_dir=None, name=None, epoch=None):
    path = os.path.curdir

    if output_dir is not None:
        path = os.path.join(path, output_dir)

    if name is not None:
        path = os.path.join(path, name)

    if epoch is not None:
        path = os.path.join(path, '{:02d}'.format(epoch))

    os.makedirs(path, exist_ok=True)

    torch.save({
        'state': G.state_dict(),
        'optimizer': g_opt.state_dict()
    }, os.path.join(path, 'generator.pth'))

    disc_dict = {
        'state': D.state_dict()
    }
    if d_opt is not None:
        disc_dict['otimizer'] = d_opt.state_dict()

    torch.save(disc_dict, os.path.join(path, 'discriminator.pth'))

    json.dump(stats, open(os.path.join(path, 'stats.json'), 'w'), indent=2)

    print('\t> Saved checkpoint checkpoint to {}'.format(path))

    return path


def checkpoint_images(images, grid_only=True, epoch=None, output_dir=None, name=None):
    path = os.path.curdir

    if output_dir is not None:
        path = os.path.join(path, output_dir)

    if name is not None:
        path = os.path.join(path, name)

    path = os.path.join(path, 'gen')

    if epoch is not None:
        path = os.path.join(path, '{:02d}'.format(epoch))

    os.makedirs(path, exist_ok=True)

    if not grid_only:
        for i, img in enumerate(images, 1):
            vutils.save_image(img, os.path.join(path, '{:02d}.png'.format(i)))

    grid = vutils.make_grid(images, padding=2, normalize=True)
    vutils.save_image(grid, os.path.join(path, 'grid.png'))
