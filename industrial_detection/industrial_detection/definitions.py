from typing import NamedTuple

import numpy as np


class Box2D(NamedTuple):
    """Holds 2D detection results.

    Attributes:
        boxes: bounding boxes in (xmin, ymin, xmax, ymax) format. Shape: [N,4]
        scores: confidence scores. Shape: [N,]
        labels: detected labels. Shape: [N,]
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    @property
    def boxes_centers(self) -> np.ndarray:
        """Calculates the center coordinates of the bounding boxes.

        Returns:
            np.ndarray: An array of shape (N, 2) containing the (x, y) center coordinates
            for each bounding box in `self.boxes`.
        """
        centers = np.hstack(
            [
                self.boxes[:, [0, 2]].mean(axis=1, keepdims=True),
                self.boxes[:, [1, 3]].mean(axis=1, keepdims=True),
            ]
        )
        return centers
