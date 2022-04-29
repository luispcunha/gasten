import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_channels, nf, num_classes):
        super(Classifier, self).__init__()

        if nf == 1:
            self.main = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_channels * 28 * 28, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
            )
        else:
            self.main = nn.Sequential(
                # (b, nc, 28, 28)
                nn.Conv2d(num_channels, nf, 5, 2, 2, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.3),
                ## (b, nf, 28, 28)
                #nn.Conv2d(nf, nf * 2, 5, 2, 2, bias=False),
                #nn.LeakyReLU(inplace=True),
                #nn.Dropout(0.3),
                ## (b, nf * 2, 28, 28)
                nn.Flatten(),
                nn.Linear(7 * 7 * nf * 4, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
                # (b, 1)
            )

    def forward(self, x):
        output = self.main(x)
        return output.flatten() if output.shape[1] == 1 else output
