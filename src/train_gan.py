import torch
import torchvision.utils as vutils
from checkpoint_utils import checkpoint_gan, checkpoint_images


def loss_terms_to_str(loss_items):
    result = ''
    for key, value in loss_items.items():
        result += '%s: %.4f ' % (key, value)

    return result


def train(config, dataset, device, n_epochs, batch_size, G, g_opt, g_crit, D, d_opt, d_crit,
          checkpoint_dir=None, checkpoint_every=1, fixed_noise=None, verbose=True):
    fixed_noise = torch.randn(64, G.nz, 1, 1, device=device) if fixed_noise is None else fixed_noise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    images = []
    stats = {
        "G_losses_epoch": [],
        "G_losses": [],
        "D_losses_epoch": [],
        "D_losses": [],
        "D_x": [],
        "D_G_z1": [],
        "D_G_z2": [],
        "D_x_epoch": [],
        "D_G_z1_epoch": [],
        "D_G_z2_epoch": [],
        "D_acc_real": [],
        "D_acc_fake_1": [],
        "D_acc_fake_2": [],
        "D_acc_real_epoch": [],
        "D_acc_fake_1_epoch": [],
        "D_acc_fake_2_epoch": [],
    }

    for loss_term in g_crit.get_loss_terms():
        stats[loss_term] = []
        stats['{}_epoch'.format(loss_term)] = []

    running_stats = {}

    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
    img = vutils.make_grid(fake, padding=2, normalize=True)
    images.append(img)

    # Storing state before starting training
    latest_cp = checkpoint_gan(G, D, g_opt, d_opt, stats, config, epoch=0, output_dir=checkpoint_dir)
    checkpoint_images(fake, 0, output_dir=checkpoint_dir)

    if verbose:
        print("Starting training loop...")
    for epoch in range(n_epochs):
        if verbose:
            print("\t> Epoch {}".format(epoch + 1))

        for loss_term in g_crit.get_loss_terms():
            running_stats[loss_term] = 0

        running_D_x = 0
        running_D_G_z1 = 0
        running_D_G_z2 = 0
        running_G_loss = 0
        running_D_loss = 0
        running_D_acc_real = 0
        running_D_acc_fake_1 = 0
        running_D_acc_fake_2 = 0

        for i, data in enumerate(dataloader, 0):
            ###
            # Discriminator
            ###
            D.zero_grad()

            # Real data batch
            real_data = data[0].to(device)
            d_output_real = D(real_data).view(-1)
            D_x = d_output_real.mean().item()
            running_D_x += d_output_real.sum().item()
            correct = (d_output_real >= 0.5).type(torch.float)
            stats["D_acc_real"].append(correct.mean(dim=0).item())
            running_D_acc_real += correct.sum(dim=0).item()
            stats["D_x"].append(D_x)

            # Fake data batch
            noise = torch.randn(data[0].shape[0], G.nz, 1, 1, device=device)
            fake_data = G(noise)
            d_output_fake = D(fake_data.detach()).view(-1)
            D_G_z1 = d_output_fake.mean().item()
            running_D_G_z1 += d_output_fake.sum().item()
            correct = (d_output_fake < 0.5).type(torch.float)
            stats["D_acc_fake_1"].append(correct.mean(dim=0).item())
            stats["D_G_z1"].append(D_G_z1)
            running_D_acc_fake_1 += correct.sum(dim=0).item()

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
            running_D_G_z2 += output.sum().item()
            correct = (output < 0.5).type(torch.float)
            stats["D_acc_fake_2"].append(correct.mean(dim=0).item())
            stats["D_G_z2"].append(D_G_z2)
            running_D_acc_fake_2 += correct.sum(dim=0).item()

            # Compute loss, gradients and update net
            g_loss, g_loss_terms = g_crit(device, output, fake_data)
            g_loss.backward()
            g_opt.step()

            for loss_term_name, loss_term_value in g_loss_terms.items():
                running_stats[loss_term_name] += loss_term_value * data[0].shape[0]
                stats[loss_term_name].append(loss_term_value)

            stats['G_losses'].append(g_loss.item())
            stats['D_losses'].append(d_loss.item())
            running_G_loss += g_loss.item() * data[0].shape[0]
            running_D_loss += d_loss.item() * data[0].shape[0]

            # Output training stats
            if verbose and ((i + 1) % 50) == 0 or i + 1 == len(dataloader):
                print('[%d/%d][%d/%d]\tD loss: %.4f\tG loss: %.4f %s\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(),
                         loss_terms_to_str(g_loss_terms), D_x, D_G_z1, D_G_z2))

        # Epoch metrics
        stats["D_x_epoch"].append(running_D_x / len(dataset))
        stats["D_G_z1_epoch"].append(running_D_G_z1 / len(dataset))
        stats["D_G_z2_epoch"].append(running_D_G_z2 / len(dataset))
        stats["G_losses_epoch"].append(running_G_loss / len(dataset))
        stats["D_losses_epoch"].append(running_D_loss / len(dataset))
        stats["D_acc_real_epoch"].append(running_D_acc_real / len(dataset))
        stats["D_acc_fake_1_epoch"].append(running_D_acc_fake_1 / len(dataset))
        stats["D_acc_fake_2_epoch"].append(running_D_acc_fake_2 / len(dataset))
        for loss_term in g_crit.get_loss_terms():
            stats['{}_epoch'.format(loss_term)].append(running_stats[loss_term] / len(dataset))

        # Save G's output on fixed noise to analyse its evolution
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2, normalize=True)
        images.append(img)

        if ((epoch + 1) % checkpoint_every) == 0:
            latest_cp = checkpoint_gan(G, D, g_opt, d_opt, stats, config, epoch=epoch+1, output_dir=checkpoint_dir)
            checkpoint_images(fake, epoch + 1, output_dir=checkpoint_dir)

    return stats, images, latest_cp
