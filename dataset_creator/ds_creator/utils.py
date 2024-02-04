import glob
import os
import shutil
from itertools import chain
from typing import Sequence

import cv2
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
