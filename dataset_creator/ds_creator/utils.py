import glob
import os
import shutil
from itertools import chain

import cv2
import numpy as np
from tqdm import tqdm


def load_image(file_path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if shape is not None:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)

    return image


def clear_directory(directory: str, clear_search: bool = False, clear_patterns: list[str] = []):
    print("\n>> Running 'clear_directory'")

    if clear_search:
        shutil.rmtree(directory, ignore_errors=True)

    elif clear_patterns:
        files_to_clean = [glob.glob(f"{directory}/**/*{pattern}", recursive=True) for pattern in clear_patterns]

        for file_path in tqdm(sorted(chain.from_iterable(files_to_clean)), desc="Cleaning metadata"):
            os.remove(file_path)
