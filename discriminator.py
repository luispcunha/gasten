import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # (b, nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, ndf * 2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, ndf * 4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (b, ndf * 8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # (b, 1)
        )

    def forward(self, x):
        return self.main(x)
