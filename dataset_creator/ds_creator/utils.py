import glob
import os
import shutil
import textwrap
from itertools import chain
from typing import Sequence

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

matplotlib.use("Agg")

OUTPUT_ARTIFACT_PATTERNS = ("_metadata.json", "_caption*", "_box2*", "_semseg*", "_instseg*", "_depth*")


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


def draw_text_as_image(text: str, save_path: str, image_size=(128, 128)):
    # wrap text to max size
    paragraph = textwrap.wrap(text, width=image_size[0] // 6)

    # create white canvas
    image = Image.new("L", image_size, color=255)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    # Create a new font with the desired size
    cur_y, pad = (image_size[1] // 5), 10
    for line in paragraph:
        text_width = draw.textlength(text=line, font=font)
        text_height = 10
        cur_x = (image_size[0] - text_width) // 2
        draw.text((cur_x, cur_y), line, font=font, fill=0)
        cur_y += text_height + pad

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


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


def gather_images_and_labels(directory: str, label_patterns: list[str]) -> dict[str, list[str]]:
    collected = {
        "images": glob_images(directory),
    }
    for pattern in label_patterns:
        collected[pattern] = glob2(directory, match_patterns=(pattern,), ignore_patterns=())

    return collected
