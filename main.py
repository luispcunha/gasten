import argparse
import torch
import torch.optim as optim

from utils import weights_init
from train_gan import train
from train_classifier import Classifier
from generator_smaller import GeneratorSmaller
from discriminator_smaller import DiscriminatorSmaller
from datasets import get_mnist
from binary_dataset import BinaryDataset
from loss import DiscriminatorLoss, GeneratorLoss, NewGeneratorLoss
from checkpoint_utils import construct_classifier_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', default='data', help='Path to dataset')
    parser.add_argument('--out-dir', dest='out_dir', default='out', help='Path to generated files')
    parser.add_argument('--name', dest='name', default='gan', help='Model name')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=2)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--image-size', dest='image_size', type=int, default=28, help='Height / width of the images')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for ADAM optimizer, default=0.5')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    full_dataset, nc = get_mnist(args.data_dir, args.image_size)

    # Create the generator
    G = GeneratorSmaller(nc, ngf=args.ngf, nz=args.nz).to(device)
    D = DiscriminatorSmaller(nc, ndf=args.ndf).to(device)

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    G.apply(weights_init)
    D.apply(weights_init)

    print(G)
    print(D)

    dataset = BinaryDataset(full_dataset, 7, 1)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Setup Adam optimizers for both G and D
    g_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    g_crit = GeneratorLoss()
    d_crit = DiscriminatorLoss()

    train(dataloader, device, args.nz, args.epochs, args.batch_size, G, g_opt, g_crit, D, d_opt=d_opt, d_crit=d_crit,
        checkpoint_dir=args.out_dir, name=args.name)

    cp = construct_classifier_from_checkpoint('out/classifier.7v1.pth', device=device)
    C = cp[0]
    C.eval()

    new_G = G

    g_crit_new = NewGeneratorLoss(C)
    g_opt_new = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    train(dataloader, device, args.nz, args.epochs, args.batch_size, new_G, g_opt_new, g_crit_new, D,
        checkpoint_dir=args.out_dir, name="fase_2")


if __name__ == '__main__':
    main()
