import numpy as np
import numpy.testing as npt
import pytest

from traffic_detection.utils.box2d import (
    compute_box_center_and_dimensions_to_xyxy,
    compute_boxes_centers_bottom_from_boxes_xyxy,
    compute_boxes_centers_from_boxes_xyxy,
)


@pytest.mark.parametrize(
    ("boxes", "expected"),
    [
        # empty input: yields empty centers
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        ),
        # single box: center is the midpoint of x1/x2 and y1/y2
        (
            np.array([[10, 20, 30, 50]], dtype=np.float32),
            np.array([[20.0, 35.0]], dtype=np.float32),
        ),
        # two boxes: vectorized computation over first dimension
        (
            np.array([[10, 20, 30, 50], [0, 0, 100, 100]], dtype=np.float32),
            np.array([[20.0, 35.0], [50.0, 50.0]], dtype=np.float32),
        ),
    ],
)
def test_compute_boxes_centers_from_boxes_xyxy(boxes: np.ndarray, expected: np.ndarray) -> None:
    """Centers are computed as mean over x-pair and y-pair for xyxy boxes."""
    out = compute_boxes_centers_from_boxes_xyxy(boxes)
    assert out.dtype == np.float32
    npt.assert_allclose(out, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("boxes", "expected"),
    [
        # empty input: yields empty center-bottoms
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        ),
        # single box: x is midpoint, y is bottom edge (y2)
        (
            np.array([[10, 20, 30, 50]], dtype=np.float32),
            np.array([[20.0, 50.0]], dtype=np.float32),
        ),
        # two boxes: vectorized computation
        (
            np.array([[10, 20, 30, 50], [0, 0, 100, 100]], dtype=np.float32),
            np.array([[20.0, 50.0], [50.0, 100.0]], dtype=np.float32),
        ),
    ],
)
def test_compute_boxes_centers_bottom_from_boxes_xyxy(boxes: np.ndarray, expected: np.ndarray) -> None:
    """Center-bottom uses x-mid and bottom y (y2) of each xyxy box."""
    out = compute_boxes_centers_bottom_from_boxes_xyxy(boxes)
    assert out.dtype == np.float32
    npt.assert_allclose(out, expected, rtol=0, atol=1e-5)


def test_compute_box_center_and_dimensions_to_xyxy_scalar() -> None:
    """Scalar center, width, height should convert to a single xyxy box."""

    # GIVEN a single box center and dimensions (w, h)
    center = np.array([20.0, 30.0], dtype=np.float32)
    w = np.array(10.0, dtype=np.float32)
    h = np.array(20.0, dtype=np.float32)

    # WHEN converting to xyxy
    xyxy = compute_box_center_and_dimensions_to_xyxy(center, w, h)
    assert xyxy.dtype == np.float32
    assert xyxy.shape == (4,)  # shape (4,) for scalar inputs

    # THEN the result should be a xyxy box with expected values
    npt.assert_allclose(xyxy, np.array([15.0, 20.0, 25.0, 40.0], dtype=np.float32), rtol=0, atol=1e-5)


def test_compute_box_center_and_dimensions_to_xyxy_vectorized_1d() -> None:
    """Vectorized center+size arrays should broadcast to (4, N) xyxy output."""

    # GIVEN two centers and two widths/heights (1-D arrays)
    centers = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    w = np.array([10.0, 20.0], dtype=np.float32)
    h = np.array([4.0, 6.0], dtype=np.float32)

    # WHEN converting to xyxy using broadcasting
    xyxy = compute_box_center_and_dimensions_to_xyxy(centers, w, h)
    # vectorized inputs yield shape (4, N)
    assert xyxy.shape == (4, 2)

    # THEN we get stacked x1,y1,x2,y2 rows with expected values per column (per box)
    expected = np.array(
        [
            [5.0, 10.0],  # x1
            [8.0, 17.0],  # y1
            [15.0, 30.0],  # x2
            [12.0, 23.0],  # y2
        ],
        dtype=np.float32,
    )
    npt.assert_allclose(xyxy, expected, rtol=0, atol=1e-5)
