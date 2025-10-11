from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_detection.definitions import Box2D
from traffic_detection.tracking.kalman import KalmanFilter2D
from traffic_detection.utils.box2d import (
    compute_box_center_and_dimensions_to_xyxy,
    compute_boxes_centers_from_boxes_xyxy,
)
from traffic_detection.utils.iou import iou_score


class Track:
    """Represents a single track in the SORT algorithm."""

    def __init__(
        self,
        box: np.ndarray,
        track_id: int,
        score: float = 0.0,
        label: str = "",
        color: str | None = None,
        bev_pos: tuple[float, float] | None = None,
        dt: float = 1.0,
    ) -> None:
        """Initialize a track with a bounding box, ID, and other parameters.

        Args:
            box: Bounding box in the format [xmin, ymin, xmax, ymax].
            track_id: Unique identifier for the track.
            label: Label associated with the track (e.g., object class).
            score: Confidence score of the detection.
            color: color label for the object.
            bev_pos: BEV position of the object.
            dt: Time step for the Kalman filter predictions.
        """
        self.kf = KalmanFilter2D(dt=dt)
        box_center = compute_boxes_centers_from_boxes_xyxy(box)
        self.kf.initiate(box_center)
        self.box = box.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.track_id = track_id
        self.age = 0
        self.time_since_update = 0
        self.hits = 1

        self.label = label
        self.score = score
        self.color: str | None = color
        self.bev_pos: tuple[float, float] | None = bev_pos

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next state of the track using the Kalman filter."""
        pred_center, velocity = self.kf.predict()
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        self.box = compute_box_center_and_dimensions_to_xyxy(pred_center, w, h)
        self.velocity = velocity
        self.age += 1
        self.time_since_update += 1
        return self.box, self.velocity

    def update(
        self,
        box: np.ndarray,
        label: str,
        score: float,
        color: str | None = None,
        bev_pos: tuple[float, float] | None = None,
    ) -> None:
        """Update the track with a new detection."""
        center = compute_boxes_centers_from_boxes_xyxy(box)
        self.kf.update(center)
        self.box = box.copy()
        self.time_since_update = 0
        self.hits += 1

        # update extra attributes
        # TODO: average out box attributes across multiple predictions.
        self.label = label
        self.score = score
        self.color = color
        self.bev_pos = bev_pos


# TODO: test other trackers:
#       - ByteTrack: Multi-Object Tracking by Associating Every Detection Box
#       - DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric


# TODO: Use a different metric for association, to deal with small objects (IoU is not good because of small overlap)
#       - e.g. distance between box centers, object appearance, etc...
#       - or at least make IoU threshold dependent on box size/class
class Sort:
    """Simple Online and Realtime Tracking (SORT) implementation.

    Tracker uses a Kalman filter for state estimation and the Hungarian algorithm for Multi Object data association.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_age: int = 3,
        min_age_predict: int = 10,
        dt: float = 1.0,
    ) -> None:
        """Initialize SORT tracker.

        Args:
            iou_threshold: IoU threshold for matching detections to tracks.
            max_age: Maximum number of frames a track can be inactive before being removed.
            min_age: Minimum age of a track to be included in the output.
            min_age_predict: Minimum number of frames to predict future state.
            dt: Time step for the Kalman filter predictions.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_age = min_age
        self.min_age_predict = min_age_predict
        self.tracks: list[Track] = []
        self.next_id = 1
        self.dt = dt

    def update(self, detections: Box2D) -> Box2D:
        """Update the tracker with new detections and output only tracked detections."""
        self._predict_all()
        matches, _, unmatched_dets = self._associate_detections_to_tracks(detections.boxes)
        self._update_matched_tracks(matches, detections)
        self._create_new_tracks(unmatched_dets, detections)
        self._prune_old_tracks()
        output_tracks = [trk for trk in self.tracks if trk.age >= self.min_age]
        return self._build_output(output_tracks)

    def _predict_all(self) -> None:
        for track in self.tracks:
            track.predict()

    def _associate_detections_to_tracks(
        self, det_boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        num_dets = det_boxes.shape[0]
        if len(self.tracks) == 0 or num_dets == 0:
            return [], [], list(range(num_dets))

        trk_boxes = np.stack([trk.box for trk in self.tracks], axis=0)
        iou_matrix = np.zeros((len(trk_boxes), num_dets), dtype=np.float32)
        for t, trk_box in enumerate(trk_boxes):
            iou_matrix[t, :] = iou_score(trk_box, det_boxes)

        matched_indices = self._linear_assignment(-iou_matrix)
        unmatched_trks = list(set(range(len(self.tracks))) - set(matched_indices[:, 0]))
        unmatched_dets = list(set(range(num_dets)) - set(matched_indices[:, 1]))

        matches: list[tuple[int, int]] = []
        for t, d in matched_indices:
            if iou_matrix[t, d] >= self.iou_threshold:
                matches.append((t, d))
            else:
                unmatched_trks.append(t)
                unmatched_dets.append(d)
        return matches, unmatched_trks, unmatched_dets

    def _extract_det_attrs(
        self,
        detections: Box2D,
        idx: int,
    ) -> tuple[str, float, str | None, tuple[float, float] | None]:
        label = str(detections.labels[idx])
        score = float(detections.scores[idx])
        color = str(detections.colors[idx]) if detections.colors is not None else None
        bev_pos = (
            (round(float(detections.bev_pos[idx][0]), 2), round(float(detections.bev_pos[idx][1]), 2))
            if detections.bev_pos is not None
            else None
        )
        return label, score, color, bev_pos

    def _update_matched_tracks(self, matches: list[tuple[int, int]], detections: Box2D) -> None:
        for t, d in matches:
            label, score, color, bev_pos = self._extract_det_attrs(detections, d)
            self.tracks[t].update(detections.boxes[d], label=label, score=score, color=color, bev_pos=bev_pos)

    def _create_new_tracks(self, unmatched_dets: list[int], detections: Box2D) -> None:
        for d in unmatched_dets:
            label, score, color, bev_pos = self._extract_det_attrs(detections, d)
            self.tracks.append(
                Track(
                    detections.boxes[d],
                    self.next_id,
                    label=label,
                    score=score,
                    color=color,
                    bev_pos=bev_pos,
                    dt=self.dt,
                )
            )
            self.next_id += 1

    def _prune_old_tracks(self) -> None:
        self.tracks = [trk for trk in self.tracks if trk.time_since_update < self.max_age]

    def _build_output(self, output_tracks: list[Track]) -> Box2D:
        if len(output_tracks) == 0:
            return Box2D.dummy(num_boxes=0)

        out_boxes = np.stack([trk.box for trk in output_tracks], axis=0, dtype=np.float32)
        out_scores = np.array([trk.score for trk in output_tracks], dtype=np.float32)
        out_labels = np.array([trk.label for trk in output_tracks], dtype=np.str_)
        out_bev_pos = np.array([trk.bev_pos for trk in output_tracks], dtype=np.float32)
        out_track_ids = np.array([trk.track_id for trk in output_tracks], dtype=np.int32)
        out_track_ages = np.array([trk.age for trk in output_tracks], dtype=np.int32)
        out_vel = np.array([trk.velocity for trk in output_tracks], dtype=np.float32)

        any_color = any((trk.color or "") != "" for trk in output_tracks)
        out_colors = np.array([trk.color or "" for trk in output_tracks], dtype=np.str_) if any_color else None

        return Box2D(
            boxes=out_boxes,
            scores=out_scores,
            labels=out_labels,
            colors=out_colors,
            bev_pos=out_bev_pos,
            track_ids=out_track_ids,
            track_ages=out_track_ages,
            vel=out_vel,
        )

    @staticmethod
    def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
        """Solve the linear assignment problem using the Hungarian algorithm."""
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

    def predict_future_state(self, future_time: float = 1.0) -> Box2D:
        """Predict the next box states for all active tracks for a given number of steps.

        Args:
            future_time: Seconds into the future to predict the state of the boxes.

        Returns:
            A Box2D object containing the predicted bounding boxes for all tracks
            at the last timestep.
        """
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_pred_bev_pos = []
        all_pred_track_ids = []
        all_pred_track_ages = []

        steps = int(future_time / self.dt)

        for tidx, track in enumerate(self.tracks):
            if track.age < self.min_age_predict:
                # Skip tracks that are too young to predict reliably
                score = 0.0
                pred_box = track.box.copy()

            else:
                original_track = deepcopy(track)

                # Predict for the specified number of steps
                score = track.score
                for _ in range(steps):
                    pred_box, _ = track.predict()

                # restore the original track state
                self.tracks[tidx] = original_track

            # Only keep the final prediction
            all_pred_boxes.append(pred_box)
            all_pred_scores.append(score)
            all_pred_labels.append(track.label)
            all_pred_bev_pos.append(track.bev_pos)
            all_pred_track_ids.append(track.track_id)
            all_pred_track_ages.append(track.age)

        # Create and return a Box2D object
        return Box2D(
            boxes=np.array(all_pred_boxes, dtype=np.float32),
            scores=np.array(all_pred_scores, dtype=np.float32),
            labels=np.array(all_pred_labels, dtype=np.str_),
            bev_pos=np.array(all_pred_bev_pos, dtype=np.float32),
            track_ids=np.array(all_pred_track_ids, dtype=np.int32),
            track_ages=np.array(all_pred_track_ages, dtype=np.int32),
        )
