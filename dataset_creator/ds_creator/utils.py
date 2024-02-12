import glob
import os
import shutil
from itertools import chain
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

OUTPUT_ARTIFACT_PATTERNS = ("_metadata.json", "_box2*", "_semseg*", "_instseg*", "_depth*")


def load_image(file_path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if shape is not None:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)

    return image


def glob2(
    directory: str,
    match_patterns: Sequence[str],
    ignore_patterns: Sequence[str] = (),
) -> list[str]:
    matches = chain.from_iterable(
        [glob.glob(f"{directory}/**/*{pattern}", recursive=True) for pattern in match_patterns]
    )
    ignored = chain.from_iterable(
        [glob.glob(f"{directory}/**/*{pattern}", recursive=True) for pattern in ignore_patterns]
    )
    selected = sorted(list(set(matches) - set(ignored)))
    return selected


def glob_images(directory: str) -> list[str]:
    return glob2(directory, match_patterns=["*.png", "*.jpeg"], ignore_patterns=OUTPUT_ARTIFACT_PATTERNS)


def clear_directory(directory: str, clear_search: bool = False, clear_patterns: Sequence[str] = ()):
    print("\n>> Running 'clear_directory'")

    if clear_search:
        shutil.rmtree(directory, ignore_errors=True)

    elif clear_patterns:
        files_to_clean = glob2(directory, match_patterns=clear_patterns)
        for file_path in tqdm(files_to_clean, desc="Cleaning metadata"):
            os.remove(file_path)


def save_images_in_grid(
    images_or_directory: list[str] | str,
    cols: int,
    save_path: str,
    title: str | None,
) -> None:

    if isinstance(images_or_directory, str) and os.path.isdir(images_or_directory):
        images = glob_images(images_or_directory)
    else:
        assert isinstance(images_or_directory, list)
        images = images_or_directory

    num_images = len(images)

    # Calculate the number of rows required
    rows = num_images // cols
    if num_images % cols != 0:
        rows += 1

    # Flatten the axes array
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=True)
    axes = axes.flatten()
    for idx in range(rows * cols):
        if idx < num_images:
            image = load_image(images[idx])
            axes[idx].imshow(image)
            axes[idx].axis("off")
            axes[idx].set_aspect("auto")

            # axes[idx].set_aspect("equal")  # Set the aspect ratio of each subplot to be equal
        else:
            axes[idx].axis("off")  # Remove empty plots for leftover squares

    fig.subplots_adjust(wspace=0, hspace=0)

    if title:
        if os.path.isfile(title):
            # Display image title instead of the text title
            img_title = load_image(title)
            fig.subplots_adjust(top=0.85)
            fig.suptitle("Image query: ")
            img_ax = fig.add_axes([0.1, 0.86, 0.8, 0.1])
            img_ax.imshow(img_title)
            img_ax.axis("off")
        else:
            # Display the text title
            fig.suptitle(title, fontsize=cols * 5, y=0.97)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved grid image to: {save_path}")


def run():
    class_images_dict = {
        "birds": glob_images("./dataset_creator/data/images/birds"),
        "cats": glob_images("./dataset_creator/data/images/cats"),
        "dogs": glob_images("./dataset_creator/data/images/dogs"),
        "humans": glob_images("./dataset_creator/data/images/humans"),
        "mixed": glob_images("./dataset_creator/data/images/mixed"),
        "cars": glob_images("./dataset_creator/data/images/cars"),
    }

    for class_name, image_list in class_images_dict.items():
        save_images_in_grid(
            image_list,
            cols=4,
            save_path=f"/home/palermo/workspace/dataset_creator/results/input_data_grid_{class_name}.png",
            title=class_name.capitalize(),
            # title="/home/palermo/workspace/dataset_creator/data/images/birds/b86ab31a85c9d98991b99dd73283326d.png",
        )
