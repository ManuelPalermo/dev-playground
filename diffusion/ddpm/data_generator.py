import os

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2 as transforms_v2


class CustomDataset(Dataset):
    def __init__(self, root: str, transforms):
        image_paths = [
            file
            for file in os.listdir(root)
            if file.endswith(".png") or file.endswith(".jpeg")
        ]
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self.transforms(image)
        label = torch.zeros(0)  # dummy label to comply with interface
        return image, label


def get_dataset(
    dataset_name="custom", directory: str = "./data", shape: tuple[int, ...] = (32, 32)
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

    # elif dataset_name == "Flowers":
    #    dataset = datasets.ImageFolder(root=dataset_root, transform=transforms)

    elif dataset_name == "custom":
        dataset = CustomDataset(root=dataset_root, transforms=transforms)
        num_classes = 1

    else:
        raise NotImplementedError(f"Unknown dataset name: {dataset_name}")

    return dataset, num_classes


def get_dataloader(
    dataset_name="custom",
    directory: str = "./data",
    data_shape: tuple[int, ...] = (3, 32, 32),
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
