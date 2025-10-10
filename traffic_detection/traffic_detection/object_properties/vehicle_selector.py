import numpy as np

from traffic_detection.definitions import Box2D
from traffic_detection.object_properties.box_filters import filter_boxes_by_property, filter_boxes_outside_polygons


class VehiclesOfInterestSelector:
    """Class to count vehicles with desired properties passing through defined areas of interest."""

    def __init__(
        self,
        areas_of_interest_polygon: dict[str, np.ndarray],
        valid_labels: list[str] | None = None,
        valid_colors: list[str] | None = None,
        min_score: float | None = None,
        min_track_age: int | None = None,
    ) -> None:
        """Initializes the VehicleCounter with specified areas of interest.

        Args:
            areas_of_interest_polygon: Area polygons to consider for counting vehicles.
            valid_labels: List of valid labels to keep.
            valid_colors: List of valid colors to keep.
            min_score: Minimum confidence score to keep a box.
            min_track_age: Minimum track age to keep.
        """

        self.areas_of_interest_polygon = areas_of_interest_polygon

        # unique boxes to have passed through area of interest
        self._areas_of_interest_counts: dict[str, set[int]] = {area: set() for area in areas_of_interest_polygon}

        self.valid_labels = valid_labels
        self.valid_colors = valid_colors
        self.min_score = min_score
        self.min_track_age = min_track_age

    def get_areas_of_interest_counts(self) -> dict[str, set[int]]:
        """Returns the history of unique track IDs in each area of interest."""
        return self._areas_of_interest_counts

    def __call__(self, detections: Box2D) -> dict[str, Box2D]:
        """Update selector for vehicles in defined areas of interest."""

        areas_of_interest_selected: dict[str, Box2D] = {}
        for area_name, polygon in self.areas_of_interest_polygon.items():
            # Check if any detection is inside the polygon
            box2d = filter_boxes_outside_polygons(
                boxes=detections,
                valid_polygons=[polygon],
            )
            box2d = filter_boxes_by_property(
                boxes=box2d,
                valid_colors=self.valid_colors,
                valid_labels=self.valid_labels,
                min_score=self.min_score,
                min_track_age=self.min_track_age,
            )
            areas_of_interest_selected[area_name] = box2d
            assert box2d.track_ids is not None, "Track IDs must be present in detections."

            self._areas_of_interest_counts[area_name].update(box2d.track_ids if box2d.track_ids is not None else [])

        return areas_of_interest_selected
