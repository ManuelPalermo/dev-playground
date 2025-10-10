from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np

from traffic_detection.definitions import Box2D
from traffic_detection.tracking.history_and_velocity_manager import convert_vel_vector_ms_to_kmh

NORMALIZATION_FUNCTION_OPTIONS = [
    "float_to_uint8",
    "uint8_to_float",
    "min_max",
    "depth",
]
NORMALIZATION_FUNCTION_OPTIONS_TYPE = Literal["float_to_uint8", "uint8_to_float", "min_max", "depth"]

TRACKS_COLOR_MAP_GENERATOR = plt.cm.get_cmap("hsv")


def get_random_color(seed: int) -> tuple[int, int, int]:
    """Generate a consistent random color from a matplotlib colormap for a given seed."""
    np.random.seed(seed)
    random_val = float(np.random.rand())
    color = TRACKS_COLOR_MAP_GENERATOR(random_val)  # Get RGBA color from colormap
    return tuple(int(c * 255) for c in color[:3])  # type: ignore[return-value]


def normalize_img(
    img: np.ndarray,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> np.ndarray:
    """Normalize an image using the specified normalization function."""
    assert normalize_func in (*NORMALIZATION_FUNCTION_OPTIONS, None)

    if normalize_func is None:
        return img

    if normalize_func == "uint8_to_float":
        return img.astype(np.float32) / 255.0

    if normalize_func == "float_to_uint8":
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if normalize_func == "min_max":
        if img.min() == img.max():
            return img.astype(np.uint8)
        return ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)

    if normalize_func == "depth":
        return (((img) / (img.max())) * 255.0).astype(np.uint8)

    raise NotImplementedError(
        f"Unknown normalization function: {normalize_func}, expected one of: {NORMALIZATION_FUNCTION_OPTIONS}"
    )


def resize_img(
    image: np.ndarray,
    size: tuple[int, int] | None = None,
    max_size: int | None = None,
    interpolation_method: int | None = None,
) -> np.ndarray:
    """Resizes and image."""

    num_channels = image.shape[-1]

    if len(image.shape) == 2 or num_channels in (1, 3):
        # its a normal image, so resize normaly with opencv
        if len(image.shape) == 2:
            image = image[..., None]

        if size:
            is_downsmaple = image.shape[0] > size[0] or image.shape[1] > size[1]
            interpolation_method = (
                interpolation_method
                if interpolation_method is not None
                else (cv2.INTER_AREA if is_downsmaple else cv2.INTER_LANCZOS4)
            )
            image = cv2.resize(image, size, interpolation=interpolation_method)

        elif max_size and (image.shape[0] > max_size or image.shape[1] > max_size):
            assert max_size
            base_width = max_size
            wpercent = base_width / float(image.shape[1])
            hsize = int(float(image.shape[0]) * float(wpercent))
            image = cv2.resize(image, (base_width, hsize), interpolation=cv2.INTER_AREA)

    else:
        # resize each channel individually
        new_img = []
        for c in range(num_channels):
            new_img.append(resize_img(image[..., c]))
        image = np.concat(new_img, axis=-1)

    return image


def load_image(
    image_path: str | Path,
    size: tuple[int, int] | None = None,
    max_size: int | None = None,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> np.ndarray:
    """Load an RGB image using OpenCV as a NumPy array (HWC, float32).

    Args:
        image_path: Path to the image file.
        size: Resize (width, height). Default: no resizing.
        max_size: Maximum size for the longest side. Default: no resizing.
        normalize_func: What normalization function to use.

    Returns:
        The image as a float32 NumPy array in RGB format.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image or unsupported format: {path}")

    image = resize_img(image=image, size=size, max_size=max_size)

    image_np = image.astype(np.float32)
    image_np = normalize_img(img=image_np, normalize_func=normalize_func)
    return image_np


def save_image(
    img: np.ndarray,
    output_path: str | Path,
    normalize_func: NORMALIZATION_FUNCTION_OPTIONS_TYPE | None = None,
) -> None:
    """Save an RGB image (NumPy array, HWC, float32 or uint8) to disk using OpenCV.

    Args:
        img: Image as a NumPy array in RGB format.
        output_path: Path to save the image.
        normalize_func: What normalization function to use.
    """

    img = normalize_img(img=img, normalize_func=normalize_func)
    Path.mkdir(Path(output_path).parent, exist_ok=True, parents=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved img to: {output_path}")


def draw_boxes(
    image: np.ndarray,
    detections: Box2D,
    box_color: tuple[int, int, int] = (0, 255, 0),
    fastest_box_color: tuple[int, int, int] | None = None,
    max_track_history: int = 100,
) -> np.ndarray:
    """Draws 2D boxes on the image and saves the visualization.

    Args:
        image: Input image.
        detections: Detection results.
        box_color: color of the points.
        fastest_box_color: color of the box to highlight the fastest box.
        max_track_history: Maximum number of past points to draw for each track.
    """
    image = image.copy()

    for box_idx in range(detections.num_boxes):
        box = detections.boxes[box_idx]
        score = detections.scores[box_idx]
        label = detections.labels[box_idx]
        obj_color = detections.colors[box_idx] if (detections.colors is not None) else ""
        bev_pos = (
            (round(float(detections.bev_pos[box_idx][0]), 2), round(float(detections.bev_pos[box_idx][1]), 2))
            if (detections.bev_pos is not None)
            else (-1, -1)
        )
        track_id = detections.track_ids[box_idx] if (detections.track_ids is not None) else -1
        track_age = detections.track_ages[box_idx] if (detections.track_ages is not None) else -1
        # velocity = (
        #    (round(float(detections.vel[box_idx][0]), 2), round(float(detections.vel[box_idx][1]), 2))
        #    if (detections.vel is not None)
        #    else (-1, -1)
        # )
        velocity_bev = (
            (round(float(detections.vel_bev[box_idx][0]), 2), round(float(detections.vel_bev[box_idx][1]), 2))
            if (detections.vel_bev is not None)
            else (-1, -1)
        )
        velocity_bev_kmh = convert_vel_vector_ms_to_kmh(velocity_bev)

        # draw fastest box in another color
        if (detections.fastest_idx is not None) and (box_idx == detections.fastest_idx):
            draw_color = fastest_box_color if fastest_box_color is not None else box_color
        else:
            draw_color = box_color

        truncated_label = label[:15]
        xmin, ymin, xmax, ymax = (int(coord) for coord in box)

        # draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), draw_color, 1)

        # draw metadata text
        cv2.putText(
            image,
            f"{truncated_label}:{score:.2f} | color: {obj_color}",
            (xmin, ymin - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            draw_color,
            1,
        )
        cv2.putText(
            image,
            f"id:{track_id} | age:{track_age} ",
            (xmin, ymin - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            draw_color,
            1,
        )
        cv2.putText(
            image,
            f"{velocity_bev_kmh}km/h | {bev_pos}m",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            draw_color,
            1,
        )

        # draw 2D velocity vector if available
        if detections.vel is not None:
            image = draw_vectors2d(
                image=image,
                center_points=np.array([detections.boxes_centers[box_idx]]),
                vectors=np.array([detections.vel[box_idx]]),
                color=draw_color,
            )

        # draw 2D track history up to a certain age (number of past points)
        if detections.track_center_history is not None:
            track_color = get_random_color(track_id)

            history = detections.track_center_history[box_idx]
            num_pts = len(history)
            # Draw faded polyline
            for i in range(1, min(num_pts, max_track_history)):
                p1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                p2 = (int(history[i][0]), int(history[i][1]))
                fade = float(i) / float(num_pts)
                col = (
                    int(track_color[0] * fade),
                    int(track_color[1] * fade),
                    int(track_color[2] * fade),
                )
                cv2.line(image, p1, p2, col, 2)

    return image


def draw_keypoints(
    image: np.ndarray,
    points: np.ndarray,
    points_classes: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    marker_size: int = 10,
) -> np.ndarray:
    """Draws 2D boxes on the image and saves the visualization.

    Args:
        image: Input image.
        points: points 2d. Shape: [N, 2].
        points_classes: points classes. Shape: [N,].
        color: color of the points.
        marker_size: size of the marker to draw.
    """
    image = image.copy()

    points_classes = (
        points_classes if points_classes is not None else np.array(["" for _ in range(len(points))], dtype=np.str_)
    )
    for point, label in zip(points, points_classes):
        truncated_label = str(label)[:15]
        (x, y) = int(point[0]), int(point[1])
        cv2.drawMarker(
            image,
            (x, y),
            color=color,
            markerType=cv2.MARKER_CROSS,
            markerSize=marker_size,
            thickness=max(1, marker_size // 5),
        )
        cv2.putText(
            image,
            f"{truncated_label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return image


def draw_polygons(
    image: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draws polygons (list of points) into an image using OpenCV. Draws both the polygon points and lines connecting them."""
    image = image.copy()
    for poly in polygons:
        if len(poly) < 2:
            continue
        pts = np.array(poly, dtype=np.int32)
        # Draw lines connecting the points (closed polygon)
        cv2.polylines(image, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=1)
        # Draw points
        for pt in pts:
            cv2.circle(image, tuple(pt), radius=2, color=color, thickness=-1)
    return image


def draw_vectors2d(
    image: np.ndarray,
    center_points: np.ndarray,
    vectors: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw 2d vectors into an image."""
    vectors_image = image.copy()
    for center_point, vector in zip(center_points, vectors):
        cv2.arrowedLine(
            vectors_image,
            center_point.astype(np.int32),
            (center_point + (vector)).astype(np.int32),
            color=color,
            thickness=1,
            tipLength=0.2,
        )
    return vectors_image
