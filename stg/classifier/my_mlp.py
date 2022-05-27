import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Classifier, self).__init__()

        self.blocks = nn.ModuleList()
        block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels*28*28, num_channels*28*14),
            nn.Dropout(0.3),
        )
        self.blocks.append(block_1)

        predictor = nn.Sequential(
            nn.Linear(num_channels*28*14, 1 if num_classes ==
                      2 else num_classes),
            nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
        )
        self.blocks.append(predictor)

    def forward(self, x, output_feature_maps=False):
        intermediate_outputs = []

        for block in self.blocks:
            x = block(x)
            intermediate_outputs.append(x)

        if intermediate_outputs[-1].shape[1] == 1:
            intermediate_outputs[-1] = intermediate_outputs[-1].flatten()

        return intermediate_outputs if output_feature_maps else intermediate_outputs[-1]
