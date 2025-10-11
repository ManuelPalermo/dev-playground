import numpy as np

from traffic_detection.utils.iou import iou_score


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
