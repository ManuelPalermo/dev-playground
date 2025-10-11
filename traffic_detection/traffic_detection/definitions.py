from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from traffic_detection.utils.box2d import (
    compute_boxes_centers_bottom_from_boxes_xyxy,
    compute_boxes_centers_from_boxes_xyxy,
)


# TODO: split Box2D attributes into other classes (e.g. TrackedBox2D class and HistoryBox2D)
class Box2D(NamedTuple):
    """Holds 2D detection results.

    Attributes: (N: number of detected boxes)
        boxes: bounding boxes in (xmin, ymin, xmax, ymax) format. Shape: [N,4]
        scores: confidence scores. Shape: [N,]
        labels: detected labels. Shape: [N,]
        colors: inferred dominant color labels. Shape: [N,]
        bev_pos: BEV position of the object. Shape: [N,2].
        track_ids: tracking identifier. Shape: [N,]
        track_ages: tracking age in frames. Shape: [N,]
        vel: Box center velocity in (vx, vy) format. Shape: [N,2]
        vel_bev: Box velocity in bev world coornates (vx, vy) format. Shape: [N,2]
        track_center_history: past history of box center points. List[List[Tuple[x, y]]], oldest to new.
        track_bev_pos_history: past history of box points in BEV coordinates. List[List[Tuple[x, y]]], oldest to new.
    """

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    labels: NDArray[np.str_]
    colors: NDArray[np.str_] | None = None
    bev_pos: NDArray[np.float32] | None = None

    # track state
    track_ids: NDArray[np.int32] | None = None
    track_ages: NDArray[np.int32] | None = None
    vel: NDArray[np.float32] | None = None

    # history
    vel_bev: NDArray[np.float32] | None = None
    track_center_history: list[list[tuple[float, float]]] | None = None
    track_bev_pos_history: list[list[tuple[float, float]]] | None = None

    @property
    def num_boxes(self) -> int:  # noqa: D102
        return len(self.boxes)

    @property
    def boxes_centers(self) -> NDArray:
        """Calculates the center coordinates of the bounding boxes.

        Returns:
            NDArray: An array of shape (N, 2) containing the (x, y) center coordinates
        """
        if not self.boxes.size:  # TODO: debug boxes not being created with [N, 4] somewhere
            return np.zeros((0, 2), dtype=np.float32)

        return compute_boxes_centers_from_boxes_xyxy(self.boxes)

    @property
    def boxes_center_bottom(self) -> NDArray:
        """Calculates the center bottom coordinates of the bounding boxes.

        Returns:
            NDArray: An array of shape (N, 2) containing the (x, y) center bottom coordinates
        """
        if not self.boxes.size:
            return np.zeros((0, 2), dtype=np.float32)

        return compute_boxes_centers_bottom_from_boxes_xyxy(self.boxes)

    @property
    def fastest_idx(self) -> int | None:
        """Returns the index of the box with the highest velocity in BEV coordinates.

        Returns:
            int | None: Index of the fastest box, or None if no boxes are present.
        """
        if self.vel_bev is None or not self.vel_bev.size:
            return None

        speeds = np.linalg.norm(self.vel_bev, axis=-1)
        return np.argmax(speeds) if speeds.size > 0 else None

    @classmethod
    def dummy(cls, num_boxes: int = 0) -> "Box2D":
        """Creates a dummy Box2D object with specified number of boxes."""
        return cls(
            boxes=np.zeros((num_boxes, 4), dtype=np.float32),
            scores=np.zeros((num_boxes,), dtype=np.float32),
            labels=np.array([""] * num_boxes, dtype=np.str_),
            colors=np.array([""] * num_boxes, dtype=np.str_),
            bev_pos=np.zeros((num_boxes, 2), dtype=np.float32),
            track_ids=np.arange(num_boxes, dtype=np.int32),
            track_ages=np.zeros((num_boxes,), dtype=np.int32),
            vel=np.zeros((num_boxes, 2), dtype=np.float32),
            vel_bev=np.zeros((num_boxes, 2), dtype=np.float32),
            track_center_history=[[] for _ in range(num_boxes)],
            track_bev_pos_history=[[] for _ in range(num_boxes)],
        )
