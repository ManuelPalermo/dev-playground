import numpy as np
import numpy.testing as npt
import pytest

from traffic_detection.definitions import Box2D


@pytest.mark.parametrize(
    ("boxes", "scores", "labels", "expected_num"),
    [
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.array([], dtype=np.str_),
            0,
        ),
        (
            np.array([[10, 20, 30, 50]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array(["car"], dtype=np.str_),
            1,
        ),
        (
            np.array([[10, 20, 30, 50], [0, 0, 100, 100]], dtype=np.float32),
            np.array([0.9, 0.8], dtype=np.float32),
            np.array(["car", "truck"], dtype=np.str_),
            2,
        ),
    ],
)
def test_num_boxes(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, expected_num: int) -> None:
    """num_boxes should match input length across cases (incl. empty)."""
    b = Box2D(boxes=boxes, scores=scores, labels=labels)
    assert b.num_boxes == expected_num


@pytest.mark.parametrize(
    ("boxes", "expected_centers"),
    [
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        ),
        (
            np.array([[10, 20, 30, 50]], dtype=np.float32),
            np.array([[20.0, 35.0]], dtype=np.float32),
        ),
        (
            np.array([[10, 20, 30, 50], [0, 0, 100, 100]], dtype=np.float32),
            np.array([[20.0, 35.0], [50.0, 50.0]], dtype=np.float32),
        ),
    ],
)
def test_boxes_centers(boxes: np.ndarray, expected_centers: np.ndarray) -> None:
    """boxes_centers returns geometric centers for xyxy boxes."""
    # GIVEN xyxy boxes
    scores = np.ones((len(boxes),), dtype=np.float32)
    labels = np.array(["x"] * len(boxes), dtype=np.str_)
    b = Box2D(boxes=boxes, scores=scores, labels=labels)

    # WHEN computing centers
    # THEN we get the midpoint in x and y for each box
    npt.assert_allclose(b.boxes_centers, expected_centers, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("boxes", "expected_bottom"),
    [
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        ),
        (
            np.array([[10, 20, 30, 50]], dtype=np.float32),
            np.array([[20.0, 50.0]], dtype=np.float32),
        ),
        (
            np.array([[10, 20, 30, 50], [0, 0, 100, 100]], dtype=np.float32),
            np.array([[20.0, 50.0], [50.0, 100.0]], dtype=np.float32),
        ),
    ],
)
def test_boxes_center_bottom(boxes: np.ndarray, expected_bottom: np.ndarray) -> None:
    """boxes_center_bottom uses bottom y and mid x."""
    # GIVEN xyxy boxes
    scores = np.ones((len(boxes),), dtype=np.float32)
    labels = np.array(["x"] * len(boxes), dtype=np.str_)
    b = Box2D(boxes=boxes, scores=scores, labels=labels)

    # WHEN computing center-bottom points
    # THEN x is the midpoint and y is the bottom (y2)
    npt.assert_allclose(b.boxes_center_bottom, expected_bottom, rtol=0, atol=1e-5)


def test_fastest_idx_none_when_no_vel_or_empty() -> None:
    """fastest_idx is None if vel_bev missing or empty."""
    # GIVEN detections but no vel_bev information (or empty array)
    # No vel_bev provided
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b1 = Box2D(boxes=boxes, scores=np.array([1.0], dtype=np.float32), labels=np.array(["a"], dtype=np.str_))
    assert b1.fastest_idx is None

    # Empty vel_bev array
    b2 = Box2D(
        boxes=boxes,
        scores=np.array([1.0], dtype=np.float32),
        labels=np.array(["a"], dtype=np.str_),
        vel_bev=np.zeros((0, 2), dtype=np.float32),
    )
    # WHEN querying fastest_idx
    # THEN result is None
    assert b2.fastest_idx is None


@pytest.mark.parametrize(
    ("vel_bev", "expected_idx"),
    [
        (np.array([[0.0, 0.0]], dtype=np.float32), 0),
        (np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32), 1),
        (np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 5.1]], dtype=np.float32), 2),
    ],
)
def test_fastest_idx_with_data(vel_bev: np.ndarray, expected_idx: int) -> None:
    """fastest_idx returns argmax of vector speed norm."""
    # GIVEN per-box velocities in BEV
    n = vel_bev.shape[0]
    boxes = np.stack([np.array([0, 0, 10, 10], dtype=np.float32)] * n)
    b = Box2D(
        boxes=boxes,
        scores=np.ones((n,), dtype=np.float32),
        labels=np.array(["a"] * n, dtype=np.str_),
        vel_bev=vel_bev,
    )
    # WHEN querying fastest_idx
    # THEN we get the index of the largest L2 norm
    assert b.fastest_idx == expected_idx


@pytest.mark.parametrize("n", [0, 3])
def test_dummy_constructor_shapes_and_defaults(n: int) -> None:
    """dummy() should produce consistent shapes and zeroed defaults."""
    # GIVEN a requested number of boxes
    # WHEN constructing a dummy Box2D via the classmethod
    b = Box2D.dummy(num_boxes=n)

    # THEN all arrays have the expected shape and zero/empty defaults
    assert b.boxes.shape == (n, 4)
    assert b.scores.shape == (n,)
    assert b.labels.shape == (n,)
    assert b.colors.shape == (n,)
    assert b.bev_pos.shape == (n, 2)
    assert b.track_ids.shape == (n,)
    assert b.track_ages.shape == (n,)
    assert b.vel.shape == (n, 2)
    assert b.vel_bev.shape == (n, 2)
    assert len(b.track_center_history) == n
    assert len(b.track_bev_pos_history) == n

    npt.assert_allclose(b.boxes, 0.0)
    npt.assert_allclose(b.scores, 0.0)
    assert all(isinstance(x, np.str_) for x in b.labels)
    assert all(isinstance(x, np.str_) for x in b.colors)
    npt.assert_allclose(b.bev_pos, 0.0)
    npt.assert_array_equal(b.track_ids, np.arange(n, dtype=np.int32))
    npt.assert_allclose(b.track_ages, 0.0)
    npt.assert_allclose(b.vel, 0.0)
    npt.assert_allclose(b.vel_bev, 0.0)
