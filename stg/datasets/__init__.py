from .datasets import get_mnist, get_fashion_mnist
from .utils import BinaryDataset


dataset_2_fn = {
    'mnist': get_mnist,
    'fashion-mnist': get_fashion_mnist,
}


def valid_dataset(name):
    valid_ds = {"mnist", "fashion-mnist"}
    return name.lower() in valid_ds


def load_dataset(name, data_dir, pos_class=None, neg_class=None, train=True):
    if not valid_dataset(name):
        print("{} dataset not supported".format(name))
        exit(-1)

    get_dset_fn = dataset_2_fn[name]
    dataset = get_dset_fn(data_dir, train=train)

    image_size = tuple(dataset.data.shape[1:])
    if len(image_size) == 2:
        image_size = 1, *image_size

    num_classes = dataset.targets.unique().size()

    if pos_class is not None and neg_class is not None:
        num_classes = 2
        dataset = BinaryDataset(dataset, pos_class, neg_class)

    return dataset, num_classes, image_size
