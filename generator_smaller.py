import torch.nn as nn


class GeneratorSmaller(nn.Module):
    def __init__(self, nc, nz=100, ngf=64):
        super(GeneratorSmaller, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.first = nn.Sequential(
            # Z: (b, nz)
            nn.Linear(nz, 7 * 7 * ngf * 4, bias=False),
            nn.BatchNorm1d(7 * 7 * ngf * 4),
            nn.ReLU(True)
            # (b, ngf * 4 * 7 * 7)
        )
        self.main = nn.Sequential(
            # (b, ngf * 4, 7, 7)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (b, ngf * 2, 7, 7)
            nn.ConvTranspose2d(ngf * 2, ngf, 5, 2, 2, bias=False, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (b, ngf, 14, 14)
            nn.ConvTranspose2d(ngf, nc, 5, 2, 2, bias=False, output_padding=1),
            nn.Tanh()
            # (b, 1, 28, 28)
        )

    def forward(self, z):
        z = self.first(z.view(-1, self.nz))
        z = self.main(z.view(-1, self.ngf * 4, 7, 7))

        return z
