from stg.gan.architectures.dcgan import Generator, Discriminator
from stg.gan.loss import NS_DiscriminatorLoss, NS_GeneratorLoss, W_GeneratorLoss, WGP_DiscriminatorLoss


def construct_gan(config, img_size, device):
    # TODO:
    use_batch_norm = config["loss"]["name"] != 'wgan-gp'
    is_critic = config["loss"]["name"] == 'wgan-gp'

    G = Generator(img_size, z_dim=config['z_dim'],
                  filter_dim=config['g_filter_dim'],
                  n_blocks=config['g_num_blocks']).to(device)

    D = Discriminator(img_size,
                      filter_dim=config['d_filter_dim'],
                      n_blocks=config['d_num_blocks'],
                      use_batch_norm=use_batch_norm, is_critic=is_critic).to(device)

    return G, D


def construct_loss(config, D):
    if config["name"] == "ns":
        return NS_GeneratorLoss(), NS_DiscriminatorLoss()
    elif config["name"] == "wgan-gp":
        return W_GeneratorLoss(), WGP_DiscriminatorLoss(D, config["args"]["lambda"])
