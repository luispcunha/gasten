import torchvision


def get_mnist(dataroot, image_size=28, train=True):
    dataset = torchvision.datasets.MNIST(root=dataroot, download=True, train=train,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.Resize(image_size),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5,), (0.5,)),
                                         ]))

    return dataset
