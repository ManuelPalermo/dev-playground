import numpy as np


def iou_score(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculates the Intersection over Union (IoU) score between a box and a set of boxes.

    Args:
        box: A 1D array of shape (4,) representing a bounding box in the format [x1, y1, x2, y2].
        boxes: A 2D array of shape (N, 4) representing N bounding boxes in the same format.

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the IoU scores between `box` and each bounding box in `boxes`.
    """
    ""
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


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str] | None = None,
    iou_threshold: float = 0.5,
    *,
    class_wise: bool = True,
) -> np.ndarray:
    """Applies Non-Maximum Suppression (NMS) to 2D boxes.

    Args:
        boxes: Array of boxes (xmin, ymin, xmax, ymax), shape [N, 4].
        scores: Array of confidence scores, shape [N,].
        labels: List of labels, shape [N,]. If None, NMS is applied globally.
        iou_threshold: IoU threshold for suppression.
        class_wise: If True, NMS is applied per class; otherwise, globally.

    Returns:
        np.ndarray: Indices of boxes to keep.
    """
    boxes = np.atleast_2d(boxes)
    scores = np.atleast_1d(scores)
    if boxes.shape[0] == 0 or scores.shape[0] == 0:
        return np.array([], dtype=int)

    keep = []
    if class_wise and labels is not None:
        unique_labels = set(labels)
        for label in unique_labels:
            idxs = [i for i, label_val in enumerate(labels) if label_val == label]
            if not idxs:
                continue
            b = boxes[idxs]
            s = np.atleast_1d(scores[idxs])
            if s.shape[0] == 0:
                continue
            order = np.argsort(s)
            if order.size > 1:
                order = order[::-1]
            while order.size > 0:
                i = order[0]
                keep.append(idxs[i])
                ious = iou_score(b[i], b[order[1:]])
                inds = np.where(ious <= iou_threshold)[0]
                order = order[inds + 1]
    else:
        s = np.atleast_1d(scores)
        order = np.argsort(s)
        if order.size > 1:
            order = order[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ious = iou_score(boxes[i], boxes[order[1:]])
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]
    return np.array(keep, dtype=int)
