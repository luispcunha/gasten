from stg.gan.architectures.dcgan import Generator, Discriminator


def construct_gan(config, img_size, device):
    G = Generator(img_size, z_dim=config['z_dim'],
                  filter_dim=config['g_filter_dim'],
                  n_blocks=config['g_num_blocks']).to(device)

    D = Discriminator(img_size,
                      filter_dim=config['d_filter_dim'],
                      n_blocks=config['d_num_blocks']).to(device)

    return G, D
