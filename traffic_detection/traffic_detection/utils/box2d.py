import numpy as np


def compute_boxes_centers_from_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Calculates the center coordinates of the 2d bounding boxes.

    Args:
        boxes: An array of shape (..., 4) containing the bounding boxes in xyxy format.

    Returns:
        np.ndarray: An array of shape (..., 2) containing the (x, y) center coordinates
    """
    return np.hstack(
        [
            boxes[..., [0, 2]].mean(axis=-1, keepdims=True),
            boxes[..., [1, 3]].mean(axis=-1, keepdims=True),
        ]
    )


def compute_boxes_centers_bottom_from_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Calculates the center bottom coordinates of the 2d bounding boxes.

    Args:
        boxes: An array of shape (..., 4) containing the bounding boxes in xyxy format.

    Returns:
        np.ndarray: An array of shape (..., 2) containing the (x, y) center bottom coordinates
    """
    return np.hstack(
        [
            boxes[..., [0, 2]].mean(axis=1, keepdims=True),
            boxes[..., [3]],  # Use the bottom y-coordinate
        ]
    )


def compute_box_center_and_dimensions_to_xyxy(
    box_center: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    """Convert box center coordinates and dimensions to bounding box in (xmin, ymin, xmax, ymax) format."""
    return np.array(
        [
            box_center[..., 0] - width / 2,
            box_center[..., 1] - height / 2,
            box_center[..., 0] + width / 2,
            box_center[..., 1] + height / 2,
        ],
        dtype=np.float32,
    )
