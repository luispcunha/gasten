import torch
import wandb
import torchvision.utils as vutils
from stg.utils.checkpoint import checkpoint_gan, checkpoint_image
from stg.utils import seed_worker
from tqdm import tqdm
import math
from stg.utils import MetricsLogger, make_grid
import matplotlib.pyplot as plt


def loss_terms_to_str(loss_items):
    result = ''
    for key, value in loss_items.items():
        result += '%s: %.4f ' % (key, value)

    return result


def evaluate(G, fid_metrics, stats_logger, batch_size, test_noise, device, c_out_hist):
    # Compute evaluation metrics on fixed noise (Z) set
    training = G.training
    G.eval()

    start_idx = 0
    num_batches = math.ceil(test_noise.size(0) / batch_size)

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        real_size = min(batch_size, test_noise.size(0) - start_idx)

        batch_z = test_noise[start_idx:start_idx + real_size]

        with torch.no_grad():
            batch_gen = G(batch_z.to(device))

        for metric_name, metric in fid_metrics.items():
            metric.update(batch_gen, (start_idx, real_size))

        c_out_hist.update(batch_gen, (start_idx, real_size))

        start_idx += batch_z.size(0)

    for metric_name, metric in fid_metrics.items():
        result = metric.finalize()
        stats_logger.update_epoch(metric_name, result, prnt=True)
        metric.reset()

    c_out_hist.plot()
    stats_logger.log_plot('histogram')
    c_out_hist.reset()

    plt.clf()

    if training:
        G.train()


def train(config, dataset, device, n_epochs, batch_size, G, g_opt, g_crit, D,
          d_opt, d_crit, test_noise, fid_metrics,
          early_stop=None,  # Tuple of (key, crit)
          start_early_stop_when=None,  # Tuple of (key, crit):
          checkpoint_dir=None, checkpoint_every=1, fixed_noise=None, c_out_hist=None):

    fixed_noise = torch.randn(
        64, G.z_dim, device=device) if fixed_noise is None else fixed_noise
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    train_metrics = MetricsLogger(prefix='train', log_epoch=False)
    eval_metrics = MetricsLogger(prefix='eval')

    train_state = {
        'epoch': 0,
        'early_stop_tracker': 0,
        'best_epoch': 0,
        'best_epoch_metric': float('inf'),
    }

    early_stop_state = 2
    if early_stop is not None:
        early_stop_key, early_stop_crit = early_stop
        early_stop_state = 1
        if start_early_stop_when is not None:
            train_state['pre_early_stop_tracker'] = 0,
            train_state['pre_early_stop_metric'] = float('inf')
            pre_early_stop_key, pre_early_stop_crit = start_early_stop_when
            early_stop_state = 0

    train_metrics.add('G_loss', every_step=True)
    train_metrics.add('D_loss', every_step=True)
    train_metrics.add('D_x', every_step=True)
    train_metrics.add('D_G_z1', every_step=True)
    train_metrics.add('D_G_z2', every_step=True)
    train_metrics.add('D_acc_real', every_step=True)
    train_metrics.add('D_acc_fake_1', every_step=True)
    train_metrics.add('D_acc_fake_2', every_step=True)

    for loss_term in g_crit.get_loss_terms():
        train_metrics.add(loss_term, every_step=True)

    for metric_name in fid_metrics.keys():
        eval_metrics.add(metric_name)

    eval_metrics.add_media_metric('samples')
    eval_metrics.add_media_metric('histogram')

    with torch.no_grad():
        G.eval()
        fake = G(fixed_noise).detach().cpu()
        G.train()

    latest_cp = checkpoint_gan(
        G, D, g_opt, d_opt, {}, {}, config, epoch=0, output_dir=checkpoint_dir)
    best_cp = latest_cp

    img = make_grid(fake)
    checkpoint_image(img, 0, output_dir=checkpoint_dir)

    G.train()
    D.train()

    print("Starting training loop...")
    for epoch in range(n_epochs):
        print("\t> Epoch {}".format(epoch + 1))

        train_metrics.reset_step_metrics()

        for i, data in enumerate(dataloader, 0):
            cur_batch_size = data[0].size(0)

            ###
            # Discriminator
            ###
            D.zero_grad()

            # Real data batch
            real_data = data[0].to(device)
            d_output_real = D(real_data).view(-1)
            D_x = d_output_real.mean().item()
            correct = (d_output_real >= 0.5).type(torch.float)
            D_acc_real = correct.mean(dim=0).item()

            train_metrics.update_step("D_x", D_x, cur_batch_size)
            train_metrics.update_step("D_acc_real", D_acc_real, cur_batch_size)

            # Fake data batch
            noise = torch.randn(cur_batch_size, G.z_dim, device=device)
            fake_data = G(noise)

            d_output_fake = D(fake_data.detach()).view(-1)
            D_G_z1 = d_output_fake.mean().item()
            correct = (d_output_fake < 0.5).type(torch.float)
            D_acc_fake_1 = correct.mean(dim=0).item()

            train_metrics.update_step("D_G_z1", D_G_z1, cur_batch_size)
            train_metrics.update_step(
                "D_acc_fake_1", D_acc_fake_1, cur_batch_size)

            # Compute loss, gradients and update net
            d_loss = d_crit(device, d_output_real, d_output_fake)
            d_loss.backward()
            d_opt.step()

            ###
            # Generator
            ###
            G.zero_grad()

            output = D(fake_data).view(-1)
            D_G_z2 = output.mean().item()
            correct = (output < 0.5).type(torch.float)
            D_acc_fake_2 = correct.mean(dim=0).item()

            train_metrics.update_step("D_G_z2", D_G_z2, cur_batch_size)
            train_metrics.update_step(
                "D_acc_fake_2", D_acc_fake_2, cur_batch_size)

            # Compute loss, gradients and update net
            g_loss, g_loss_terms = g_crit(device, output, fake_data)
            g_loss.backward()
            g_opt.step()

            for loss_term_name, loss_term_value in g_loss_terms.items():
                train_metrics.update_step(
                    loss_term_name, loss_term_value, cur_batch_size)

            train_metrics.update_step('G_loss', g_loss.item(), cur_batch_size)
            train_metrics.update_step('D_loss', d_loss.item(), cur_batch_size)

            # Output training stats
            if ((i + 1) % 50) == 0 or i + 1 == len(dataloader):
                print('[%d/%d][%d/%d]\tD loss: %.4f\tG loss: %.4f %s\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(),
                         loss_terms_to_str(g_loss_terms), D_x, D_G_z1, D_G_z2))

            train_metrics.finalize_step(cur_batch_size)

        train_metrics.finalize_epoch()
        train_state['epoch'] += 1

        # Save G's output on fixed noise to analyse its evolution
        with torch.no_grad():
            G.eval()
            fake = G(fixed_noise).detach().cpu()
            G.train()

        # Compute epoch FID on fixed set
        evaluate(G, fid_metrics, eval_metrics, batch_size,
                 test_noise, device, c_out_hist)
        eval_metrics.finalize_epoch()

        img = make_grid(fake)
        checkpoint_image(img, epoch + 1, output_dir=checkpoint_dir)
        eval_metrics.log_image('samples', img)

        if ((epoch + 1) % checkpoint_every) == 0:
            latest_cp = checkpoint_gan(
                G, D, g_opt, d_opt, train_state, {"eval": eval_metrics.stats, "train": train_metrics.stats}, config, epoch=epoch + 1, output_dir=checkpoint_dir)

        # Early stop
        if early_stop_state == 0:
            # Pre early stop phase
            if eval_metrics.stats[f'eval/{pre_early_stop_key}'][-1] \
                    < train_state['pre_early_stop_metric']:
                train_state['pre_early_stop_tracker'] = 0
                train_state['pre_early_stop_metric'] = \
                    eval_metrics.stats[f'eval/{pre_early_stop_key}'][-1]
            else:
                train_state['pre_early_stop_tracker'] += 1
                print(
                    " > Pre-early stop tracker {}/{}".format(train_state['pre_early_stop_tracker'], pre_early_stop_crit))
                if train_state['pre_early_stop_tracker'] \
                        == pre_early_stop_crit:
                    early_stop_state = 1

            best_cp = latest_cp
        elif early_stop_state == 1:
            # Early stop phase
            if eval_metrics.stats[f'eval/{early_stop_key}'][-1] < train_state['best_epoch_metric']:
                train_state['early_stop_tracker'] = 0
                train_state['best_epoch'] = epoch
                train_state['best_epoch_metric'] = eval_metrics.stats[
                    f'eval/{early_stop_key}'][-1]
                best_cp = latest_cp
            else:
                train_state['early_stop_tracker'] += 1
                print(
                    " > Early stop tracker {}/{}".format(train_state['early_stop_tracker'], early_stop_crit))
                if train_state['early_stop_tracker'] == early_stop_crit:
                    break
        else:
            # No early stop
            best_cp = latest_cp

    return train_state, best_cp
