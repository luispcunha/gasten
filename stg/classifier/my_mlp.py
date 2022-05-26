import torch.nn as nn


class MyMLP(nn.Module):
    def __init__(self, nc, nf, num_classes):
        super(MyMLP, self).__init__()

        self.blocks = nn.ModuleList()
        block_1 = nn.Sequential(
            nn.Conv2d(nc, nf * 2, 5, 2, 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.blocks.append(block_1)
        block_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
        )
        self.blocks.append(block_2)
        predictor = nn.Sequential(
            nn.Linear(7 * 7 * nf * 2, 1 if num_classes == 2 else num_classes),
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
