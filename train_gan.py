import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from checkpoint_utils import checkpoint_gan, checkpoint_images


def loss_items_to_str(loss_items):
    result = ''
    for key, value in loss_items.items():
        result += '%s: %.4f ' % (key, value)

    return result


def train(dataloader, device, nz, n_epochs, batch_size, G, g_opt, g_crit, D, d_opt=None, d_crit=None,
          checkpoint_dir=None, checkpoint_every=5, name=None):
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
    img = vutils.make_grid(fake, padding=2, normalize=True)
    img_list.append(img)
    checkpoint_images(fake, epoch=0, output_dir=checkpoint_dir, name=name)

    train_d = True
    d_loss = torch.zeros((1,))
    if d_opt is None or d_crit is None:
        D.eval()
        train_d = False


    print("Starting training loop...")
    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            ###
            # Discriminator
            ###
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_data = G(noise)

            if train_d:
                D.zero_grad()

                # Fake data batch
                d_output_fake = D(fake_data.detach()).view(-1)

                D_G_z1 = d_output_fake.mean().item()

                # Real data batch
                real_data = data[0].to(device)
                d_output_real = D(real_data).view(-1)

                D_x = d_output_real.mean().item()

                # Compute loss and gradients
                d_loss = d_crit(device, d_output_real, d_output_fake)
                d_loss.backward()

                # Update D
                d_opt.step()

            ###
            # Generator
            ###
            G.zero_grad()

            output = D(fake_data).view(-1)

            D_G_z2 = output.mean().item()

            # Compute loss and gradients
            g_loss, g_loss_items = g_crit(device, output, fake_data)
            g_loss.backward()

            # Update D
            g_opt.step()

            # Output training stats
            if ((i + 1) % 50) == 0 or i + 1 == len(dataloader) :
                if train_d:
                    print('[%d/%d][%d/%d]\tD loss: %.4f\tG loss: %.4f%s\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item(),
                             loss_items_to_str(g_loss_items), D_x, D_G_z1, D_G_z2))
                else:
                    print('[%d/%d][%d/%d]\tG loss: %.4f %s\tD(G(z)): %.4f'
                          % (epoch + 1, n_epochs, i + 1, len(dataloader), g_loss.item(),
                             loss_items_to_str(g_loss_items), D_G_z2))

            # Save Losses for plotting later
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            iters += 1

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2, normalize=True)
        img_list.append(img)

        if ((epoch + 1) % checkpoint_every) == 0:
            checkpoint_gan(G, D, g_opt, d_opt, '', epoch=epoch+1, output_dir=checkpoint_dir, name=name)
            checkpoint_images(fake, epoch=epoch+1, output_dir=checkpoint_dir, name=name)

    return img_list, g_losses, d_losses
