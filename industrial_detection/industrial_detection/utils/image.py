from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from industrial_detection.definitions import Box2D

NORMALIZATION_FUNCTION_OPTIONS = [
    "float_to_uint8",
    "uint8_to_float",
    "min_max",
    "depth",
    "mask",
]
NORMALIZATION_FUNCTION_OPTIONS_TYPE = Literal["float_to_uint8", "uint8_to_float", "min_max", "depth", "mask"]


def normalize_img(
    img: np.ndarray,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> np.ndarray:
    assert normalize_func in (*NORMALIZATION_FUNCTION_OPTIONS, None)

    if normalize_func is None:
        return img

    if normalize_func == "uint8_to_float":
        return img.astype(np.float32) / 255.0

    if normalize_func == "float_to_uint8":
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if normalize_func == "min_max":
        if img.min() == img.max():
            return img.astype(np.uint8)
        return ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)

    if normalize_func == "depth":
        return (((img) / (img.max())) * 255.0).astype(np.uint8)

    if normalize_func == "mask":
        return draw_masks(
            image=np.tile(np.zeros_like(img, dtype=np.uint8)[..., None], reps=(1, 1, 3)),
            masks=img,
        )

    raise NotImplementedError(
        f"Unknown normalization function: {normalize_func}, expected one of: {NORMALIZATION_FUNCTION_OPTIONS}"
    )


def resize_img(
    image: np.ndarray,
    size: tuple[int, int] | None = None,
    max_size: int | None = None,
    interpolation_method: int | None = None,
) -> np.ndarray:
    """Resizes and image."""

    num_channels = image.shape[-1]

    if len(image.shape) == 2 or num_channels in (1, 3):
        # its a normal image, so resize normaly with opencv
        if len(image.shape) == 2:
            image = image[..., None]

        if size:
            is_downsmaple = image.shape[0] > size[0] or image.shape[1] > size[1]
            interpolation_method = (
                interpolation_method
                if interpolation_method is not None
                else (cv2.INTER_AREA if is_downsmaple else cv2.INTER_LANCZOS4)
            )
            image = cv2.resize(image, size, interpolation=interpolation_method)

        elif max_size and (image.shape[0] > max_size or image.shape[1] > max_size):
            assert max_size
            base_width = max_size
            wpercent = base_width / float(image.shape[1])
            hsize = int(float(image.shape[0]) * float(wpercent))
            image = cv2.resize(image, (base_width, hsize), interpolation=cv2.INTER_AREA)

    else:
        # resize each channel individually
        new_img = []
        for c in range(num_channels):
            new_img.append(resize_img(image[..., c]))
        image = np.concat(new_img, axis=-1)

    return image


def load_image(
    image_path: str | Path,
    size: tuple[int, int] | None = None,
    max_size: int | None = None,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> np.ndarray:
    """Load an RGB image using OpenCV as a NumPy array (HWC, float32).

    Args:
        image_path: Path to the image file.
        size: Resize (width, height). Default: no resizing.
        normalize_func: What normalization function to use.

    Returns:
        The image as a float32 NumPy array in RGB format.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image or unsupported format: {path}")

    image = resize_img(image=image, size=size, max_size=max_size)

    image_np = image.astype(np.float32)
    image_np = normalize_img(img=image_np, normalize_func=normalize_func)
    return image_np


def save_image(
    img: np.ndarray,
    output_path: str | Path,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> None:
    """Save an RGB image (NumPy array, HWC, float32 or uint8) to disk using OpenCV.

    Args:
        img: Image as a NumPy array in RGB format.
        output_path: Path to save the image.
        normalize_func: What normalization function to use.
    """

    img = normalize_img(img=img, normalize_func=normalize_func)
    Path.mkdir(Path(output_path).parent, exist_ok=True, parents=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved img to: {output_path}")


def draw_boxes(
    image: np.ndarray,
    detections: Box2D,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draws 2D boxes on the image and saves the visualization.

    Args:
        image: Input image.
        detections: Detection results.
        color: color of the points.
    """
    image = image.copy()

    for box, score, label in zip(detections.boxes, detections.scores, detections.labels):
        truncated_label = label[:15]
        xmin, ymin, xmax, ymax = (int(coord) for coord in box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            f"{truncated_label}:{score:.2f}",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
    return image


def draw_keypoints(
    image: np.ndarray,
    points: np.ndarray,
    points_classes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draws 2D boxes on the image and saves the visualization.

    Args:
        image: Input image.
        points: points 2d. Shape: [N, 2].
        points_classes: points classes. Shape: [N,].
        color: color of the points.
    """
    image = image.copy()

    for point, label in zip(points, points_classes):
        truncated_label = str(label)[:15]
        (x, y) = int(point[0]), int(point[1])
        cv2.drawMarker(image, (x, y), color=color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.putText(
            image,
            f"{truncated_label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
    return image


def draw_masks(
    image: np.ndarray,
    masks: dict[int, np.ndarray] | np.ndarray,
    alpha: float = 0.5,
    draw_border: bool = True,
) -> np.ndarray:
    """Draw masks into an image."""
    rng = np.random.default_rng(0)
    colors = rng.uniform(0, 256, size=(256, 3))

    if not isinstance(masks, dict):
        labels = np.unique(masks)
        masks_dict = {int(lbl): np.where(masks == lbl, masks, 0.0).astype(np.uint8) for lbl in labels}
        return draw_masks(image=image, masks=masks_dict)

    mask_image = image.copy()

    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = colors[label_id]
        mask_image = draw_mask(mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border)

    return mask_image


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.5,
    draw_border: bool = True,
) -> np.ndarray:
    """Draw a mask into an image."""
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image


def draw_vectors2d(image: np.ndarray, center_points: np.ndarray, vectors: np.ndarray, lenght: int = 50) -> np.ndarray:
    """Draw 2d vectors into an image."""
    vectors_image = image.copy()
    for center_point, vector in zip(center_points, vectors):
        cv2.arrowedLine(
            vectors_image,
            center_point.astype(np.int32),
            (center_point + (vector * lenght)).astype(np.int32),
            color=(0, 0, 255),
            thickness=lenght // 10,
            tipLength=0.3,
        )
    return vectors_image
