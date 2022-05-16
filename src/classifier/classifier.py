import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_channels, nf, num_classes):
        """
        output_block  if != None returns list with feature maps, final classification is in position -1
        """
        super(Classifier, self).__init__()

        self.blocks = nn.ModuleList()
        if nf == 1:
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_channels * 28 * 28, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
            )
            self.blocks.append(classifier)
        else:
            block_1 = nn.Sequential(
                nn.Conv2d(num_channels, nf, 5, 2, 2, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.blocks.append(block_1)
            block_2 = nn.Sequential(
                nn.Conv2d(nf, nf * 2, 5, 2, 2, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Flatten(),
            )
            self.blocks.append(block_2)
            classifier = nn.Sequential(
                nn.Linear(7 * 7 * nf * 2, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
            )
            self.blocks.append(classifier)

    def forward(self, x, output_feature_maps=False):
        intermediate_outputs = []

        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
