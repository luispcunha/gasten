import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, nc, nf):
        super(Classifier, self).__init__()

        if nf == 1:
            self.main = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nc * 28 * 28, 1),
                nn.Sigmoid()
            )
        else:
            self.main = nn.Sequential(
                # (b, nc, 28, 28)
                nn.Conv2d(nc, nf, 5, 2, 2, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.3),
                ## (b, nf, 28, 28)
                #nn.Conv2d(nf, nf * 2, 5, 2, 2, bias=False),
                #nn.LeakyReLU(inplace=True),
                #nn.Dropout(0.3),
                ## (b, nf * 2, 28, 28)
                nn.Flatten(),
                nn.Linear(7 * 7 * nf * 4, 1),
                nn.Sigmoid()
                # (b, 1)
            )

    def forward(self, x):
        return self.main(x).flatten()
