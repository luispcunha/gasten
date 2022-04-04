import torch.nn as nn


class DiscriminatorSmaller(nn.Module):
    def __init__(self, nc, ndf=64):
        super(DiscriminatorSmaller, self).__init__()
        self.main = nn.Sequential(
            # (b, nc, 28, 28)
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # (b, ndf, 14, 14)
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # (b, ndf * 2, 7, 7)
            nn.Flatten(),
            nn.Linear(7 * 7 * ndf * 2, 1),
            nn.Sigmoid()
            # (b, 1)
        )

    def forward(self, x):
        return self.main(x)
