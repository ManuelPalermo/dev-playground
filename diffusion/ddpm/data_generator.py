"""Dataloader and dataset utilities, supports 2D image and 3D pointcloud data."""

import os

import torch
import torch_geometric.transforms as transforms_3d
import torchvision.datasets as datasets_image
from torch.utils.data import DataLoader, Dataset
from torch_geometric import datasets as datasets_3d
from torchvision.transforms import v2 as transforms_v2

IMAGE_DATASETS = (
    "MNIST",
    "Cifar-10",
    "Cifar-100",
    "ImageFolder",
    "ImageFolder_overfit",
)

POINT_DATASETS = (
    "ModelNet10",
    "ModelNet40",
    "ShapeNet",
)


class CacheRepeatDataset(Dataset):
    """Wraps a pytorch dataset to enable caching and sampling repeated times."""

    # cache a maximum of 128 images
    CACHE_SIZE: int = 512

    def __init__(self, dataset: Dataset, repeat_num: int = 1):
        """Constructor.

        Args:
            dataset: pytorch dataset
            repeat_num: number of times to repeat samples.
        """
        self.dataset = dataset
        self.repeat_num = repeat_num

        assert (
            len(self.dataset) < CacheRepeatDataset.CACHE_SIZE
        ), "With this many images you prob will get issues with caching."

        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.dataset) * self.repeat_num

    def __getitem__(self, idx: int):
        """Function to sample from the dataset at a given idx."""
        # convert repeated idx to actual dataset idx
        orig_idx = idx % len(self.dataset)

        if orig_idx in self._cache:
            return self._cache[orig_idx]  # sample from cache
        else:
            sample = self.dataset.__getitem__(orig_idx)  # sample for original dataset
            self._cache[orig_idx] = sample  # cache sample for original dataset
            return sample


def get_dataset_image(
    dataset_name="MNIST", directory: str = "./data", shape: tuple[int, ...] = (28, 28)
) -> tuple[Dataset, int]:
    """Creates a pytorch dataset from 2D iamge data."""
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

    if dataset_name == "MNIST":
        dataset = datasets_image.MNIST(root=dataset_root, train=True, download=True, transform=transforms)
        num_classes = 10

    elif dataset_name == "Cifar-10":
        dataset = datasets_image.CIFAR10(root=dataset_root, train=True, download=True, transform=transforms)
        num_classes = 10

    elif dataset_name == "Cifar-100":
        dataset = datasets_image.CIFAR10(root=dataset_root, train=True, download=True, transform=transforms)
        num_classes = 100

    elif dataset_name == "ImageFolder" or dataset_name == "ImageFolder_overfit":
        dataset = datasets_image.ImageFolder(root=directory, transform=transforms)
        num_classes = len(dataset.classes)

        if dataset_name == "ImageFolder_overfit":
            # if you want to overfit on a small dataset (just cache samples and repeat idx to balance epoch logic)
            dataset = CacheRepeatDataset(dataset=dataset, repeat_num=2000)

    else:
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")

    return dataset, num_classes


def get_dataset_points(
    dataset_name: str = "ShapeNet", directory: str = "./data", shape: tuple[int, ...] = (3000,)
) -> tuple[Dataset, int]:
    """Creates a pytorch dataset with 3D pointcloud data."""
    dataset_root = os.path.join(directory, dataset_name)

    # op to pad number of points to max
    pad_fn = lambda p: torch.nn.functional.pad(p, pad=(0, 0, 0, shape[0] - p.shape[0]), value=0.0)
    ch_first = lambda data: torch.permute(data, dims=(1, 0))

    if dataset_name == "ShapeNet":
        # use only some easy to distinguish categories
        categories = ["Airplane", "Car", "Chair", "Motorbike", "Pistol"]

        transforms = transforms_3d.Compose([lambda data: (ch_first(pad_fn(data.pos)), torch.squeeze(data.category))])
        dataset = datasets_3d.ShapeNet(
            root=dataset_root, categories=categories, include_normals=False, transform=transforms
        )
        num_classes = len(categories) if categories is not None else dataset.num_classes

    # TODO: fix issues with ModelNet data scale (points with very large xyz values break stuff)
    elif dataset_name == "ModelNet10":
        transforms = transforms_3d.Compose([lambda data: (ch_first(pad_fn(data.pos)), torch.squeeze(data.y))])
        dataset = datasets_3d.ModelNet(root=dataset_root, name="10", train=True, transform=transforms)
        num_classes = 10

    elif dataset_name == "ModelNet40":
        transforms = transforms_3d.Compose([lambda data: (ch_first(pad_fn(data.pos)), torch.squeeze(data.y))])
        dataset = datasets_3d.ModelNet(root=dataset_root, name="40", train=True, transform=transforms)
        num_classes = 40

    else:
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")

    # image: (32, 32, 3) -> vis: (302, 32, 3)
    return dataset, num_classes


def get_dataloader(
    dataset_name="MNIST",
    directory: str = "./data",
    data_shape: tuple[int, ...] = (1, 28, 28),
    batch_size: int = 32,
    shuffle: bool = True,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, int]:
    """Creates pytorch dataloader."""
    if dataset_name in IMAGE_DATASETS:
        dataset, num_classes = get_dataset_image(dataset_name, directory, shape=data_shape[1:])
    elif dataset_name in POINT_DATASETS:
        dataset, num_classes = get_dataset_points(dataset_name, directory, shape=data_shape[1:])
    else:
        raise NotImplementedError(
            f"Unknown dataset name: {dataset_name}. Available ones are:"
            f"\n - IMAGE_DATASETS: {IMAGE_DATASETS}"
            f"\n - POINT_DATASETS: {POINT_DATASETS}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
    )
    return dataloader, num_classes
