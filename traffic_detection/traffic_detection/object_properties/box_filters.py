import numpy as np
from shapely.geometry import Point, Polygon

from traffic_detection.definitions import Box2D


def filter_boxes_outside_polygons(boxes: Box2D, valid_polygons: list[np.ndarray]) -> Box2D:
    """Filter boxes which dont have their center inside any of the valid polygons."""

    # Prepare shapely polygons
    shapely_polys = [Polygon(poly) for poly in valid_polygons]

    # Check for each box center if inside any polygon
    keep_mask = np.array(
        [any(poly.contains(Point(center)) for poly in shapely_polys) for center in boxes.boxes_centers],
        dtype=np.bool,
    )
    keep_mask_idx = np.where(keep_mask)[0]

    return Box2D(
        boxes=boxes.boxes[keep_mask],
        scores=boxes.scores[keep_mask],
        labels=boxes.labels[keep_mask],
        colors=boxes.colors[keep_mask] if boxes.colors is not None else None,
        bev_pos=boxes.bev_pos[keep_mask] if boxes.bev_pos is not None else None,
        track_ids=boxes.track_ids[keep_mask] if boxes.track_ids is not None else None,
        track_ages=boxes.track_ages[keep_mask] if boxes.track_ages is not None else None,
        vel=boxes.vel[keep_mask] if boxes.vel is not None else None,
        vel_bev=boxes.vel_bev[keep_mask] if boxes.vel_bev is not None else None,
        track_center_history=(
            [boxes.track_center_history[idx] for idx in keep_mask_idx]
            if boxes.track_center_history is not None
            else None
        ),
        track_bev_pos_history=(boxes.track_bev_pos_history if boxes.track_bev_pos_history is not None else None),
    )


def filter_boxes_by_property(
    boxes: Box2D,
    valid_labels: list[str] | None = None,
    valid_colors: list[str] | None = None,
    min_score: float | None = None,
    min_track_age: int | None = None,
) -> Box2D:
    """Filter boxes which don't have the desired properties.

    Args:
        boxes: The input Box2D object containing detection results.
        valid_labels: List of valid labels to keep.
        valid_colors: List of valid colors to keep.
        min_score: Minimum confidence score to keep a box.
        min_track_age: Minimum track age to keep.

    Returns:
        A filtered Box2D object containing only the boxes that match the desired properties.
    """
    keep_mask = np.ones(len(boxes.boxes), dtype=np.bool)

    if valid_labels is not None:
        keep_mask &= np.isin(boxes.labels, valid_labels)

    if valid_colors is not None and boxes.colors is not None:
        keep_mask &= np.isin(boxes.colors, valid_colors)

    if min_score is not None:
        keep_mask &= boxes.scores >= min_score

    if min_track_age is not None and boxes.track_ages is not None:
        keep_mask &= boxes.track_ages >= min_track_age

    keep_mask_idx = np.where(keep_mask)[0]

    return Box2D(
        boxes=boxes.boxes[keep_mask],
        scores=boxes.scores[keep_mask],
        labels=boxes.labels[keep_mask],
        colors=boxes.colors[keep_mask] if boxes.colors is not None else None,
        bev_pos=boxes.bev_pos[keep_mask] if boxes.bev_pos is not None else None,
        track_ids=boxes.track_ids[keep_mask] if boxes.track_ids is not None else None,
        track_ages=boxes.track_ages[keep_mask] if boxes.track_ages is not None else None,
        vel=boxes.vel[keep_mask] if boxes.vel is not None else None,
        vel_bev=boxes.vel_bev[keep_mask] if boxes.vel_bev is not None else None,
        track_center_history=(
            [boxes.track_center_history[idx] for idx in keep_mask_idx]
            if boxes.track_center_history is not None
            else None
        ),
        track_bev_pos_history=(boxes.track_bev_pos_history if boxes.track_bev_pos_history is not None else None),
    )
