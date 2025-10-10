import numpy as np


def iou_score(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculates the Intersection over Union (IoU) score between a box and a set of boxes.

    Args:
        box: A 1D array of shape (4,) representing a bounding box in the format [x1, y1, x2, y2].
        boxes: A 2D array of shape (N, 4) representing N bounding boxes in the same format.

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the IoU scores between `box` and each bounding box in `boxes`.
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou
