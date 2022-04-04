import torch
import torch.nn.functional as F


class DiscriminatorLoss:
    def __call__(self, device, real_output, fake_output):
        ones = torch.ones_like(real_output, dtype=torch.float, device=device)
        zeros = torch.zeros_like(fake_output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(real_output, ones) + F.binary_cross_entropy(fake_output, zeros)


class GeneratorLoss:
    def __call__(self, device, output, fake_data):
        ones = torch.ones_like(output, dtype=torch.float, device=device)

        return F.binary_cross_entropy(output, ones), {}


class NewGeneratorLossBinary(GeneratorLoss):
    def __init__(self, classifier, beta=0.5):
        self.classifier = classifier
        self.beta = beta

    def __call__(self, device, output, fake_data):
        ones = torch.ones_like(output, dtype=torch.float, device=device)
        term_1 = F.binary_cross_entropy(output, ones)

        c_output = self.classifier(fake_data)
        top_2 = torch.topk(c_output, 2, dim=1).values
        term_2 = top_2[:, 0] - top_2[:, 1]

        loss = (1 - self.beta) * term_1 + self.beta * term_2

        return loss, {'term_1': term_1.item(), 'term_2': term_2.item()}


class NewGeneratorLossBinary(GeneratorLoss):
    def __init__(self, classifier, beta=0.5):
        self.classifier = classifier
        self.beta = beta

    def __call__(self, device, output, fake_data):
        ones = torch.ones_like(output, dtype=torch.float, device=device)
        term_1 = F.binary_cross_entropy(output, ones)

        c_output = self.classifier(fake_data)
        term_2 = 2 * (0.5 - c_output).abs().mean()

        loss = (1 - self.beta) * term_1 + self.beta * term_2

        return loss, {'term_1': term_1.item(), 'term_2': term_2.item()}
