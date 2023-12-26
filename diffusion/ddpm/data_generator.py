import os

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as transforms_v2


class CacheRepeatDataset(Dataset):
    # cache a maximum of 128 images
    CACHE_SIZE: int = 512

    def __init__(self, dataset: Dataset, repeat_num: int = 1):
        self.dataset = dataset
        self.repeat_num = repeat_num

        assert (
            len(self.dataset) < CacheRepeatDataset.CACHE_SIZE
        ), "With this many images you prob will get issues with caching."

        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.dataset) * self.repeat_num

    def __getitem__(self, idx: int):
        # convert repeated idx to actual dataset idx
        orig_idx = idx % len(self.dataset)

        if orig_idx in self._cache:
            return self._cache[orig_idx]  # sample from cache
        else:
            sample = self.dataset.__getitem__(orig_idx)  # sample for original dataset
            self._cache[orig_idx] = sample  # cache sample for original dataset
            return sample


def get_dataset(
    dataset_name="MNIST", directory: str = "./data", shape: tuple[int, ...] = (28, 28)
) -> tuple[Dataset, int]:
    transforms = transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize(
                shape,
                interpolation=transforms_v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            # transforms_v2.RandomHorizontalFlip(),  # not compatible with all datasets (e.g. MNIST)
            transforms_v2.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    dataset_root = os.path.join(directory, dataset_name)

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(
            root=dataset_root, train=True, download=True, transform=transforms
        )
        num_classes = 10

    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(
            root=dataset_root, train=True, download=True, transform=transforms
        )
        num_classes = 10

    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(
            root=dataset_root, train=True, download=True, transform=transforms
        )
        num_classes = 100

    elif dataset_name == "ImageFolder" or dataset_name == "ImageFolder_overfit":
        dataset = datasets.ImageFolder(root=directory, transform=transforms)
        num_classes = len(dataset.classes)

        if dataset_name == "ImageFolder_overfit":
            # if you want to overfit on a small dataset (just cache samples and repeat idx to balance epoch logic)
            dataset = CacheRepeatDataset(dataset=dataset, repeat_num=2000)

    else:
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")

    return dataset, num_classes


def get_dataloader(
    dataset_name="MNIST",
    directory: str = "./data",
    data_shape: tuple[int, ...] = (1, 28, 28),
    batch_size: int = 32,
    pin_memory: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[Dataset, int]:
    dataset, num_classes = get_dataset(dataset_name, directory, shape=data_shape[1:])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return dataloader, num_classes
