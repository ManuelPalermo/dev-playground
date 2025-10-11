from typing import Literal

import cv2
import numpy as np

from traffic_detection.definitions import Box2D


def extract_box_patches(
    image: np.ndarray,
    boxes: np.ndarray,
    min_box_size: int = 4,
    box_offset: float = -0.2,
) -> list[np.ndarray | None]:
    """Extract image patches for each box.

    Args:
        image: Input RGB image.
        boxes: Array of boxes (N,4) in (xmin, ymin, xmax, ymax) format.
        min_box_size: Minimum width/height to accept a patch.
        box_offset: Offset as a percentage of box dimensions to expand/reduce the patch.

    Returns:
        List of patches or None where extraction is invalid.
    """
    h_img, w_img = image.shape[:2]
    patches: list[np.ndarray | None] = []
    for b in boxes.astype(int):
        x1, y1, x2, y2 = b

        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1

        # Apply offset as a percentage of box dimensions
        offset_x = int(box_width * box_offset)
        offset_y = int(box_height * box_offset)

        # Adjust box coordinates
        x1 = max(0, min(w_img - 1, x1 - offset_x))
        x2 = max(0, min(w_img - 1, x2 + offset_x))
        y1 = max(0, min(h_img - 1, y1 - offset_y))
        y2 = max(0, min(h_img - 1, y2 + offset_y))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            patches.append(None)
            continue
        patch = image[y1:y2, x1:x2]
        patches.append(patch if patch.size > 0 else None)
    return patches


class ObjectPropertiesClassification:
    """Object properties classifier on image box patches.

    Currently implemented properties:
        - Color (based on RGB or HSV color spaces)

    TODO: explore extending with a CLIP based model to infer arbitrary properties of interest (evaluate if runtime ok)

    """

    def __init__(self, min_box_size: int = 4, color_scheme: Literal["rgb", "hsv"] = "hsv") -> None:
        self.min_box_size = min_box_size

        self.color_scheme = color_scheme
        match color_scheme:
            case "rgb":
                self._infer_color_fn = self._infer_color_rgb
            case "hsv":
                self._infer_color_fn = self._infer_color_hsv
            case _:
                raise ValueError(f"Unsupported color scheme: {color_scheme}")

    def __call__(self, detections: Box2D, image: np.ndarray) -> Box2D:
        """Classify object colors based on image patches."""

        # get image box patches
        patches = extract_box_patches(image, detections.boxes, self.min_box_size)

        # infer color property for each object
        colors: list[str] = []
        for patch in patches:
            if patch is None:
                colors.append("")
                continue
            colors.append(self._infer_color_fn(patch))
        colors_arr = np.array(colors, dtype=np.str_)

        return Box2D(
            boxes=detections.boxes,
            scores=detections.scores,
            labels=detections.labels,
            colors=colors_arr,
            bev_pos=detections.bev_pos if detections.bev_pos is not None else None,
            track_ids=detections.track_ids if detections.track_ids is not None else None,
            track_ages=detections.track_ages if detections.track_ages is not None else None,
            vel=detections.vel if detections.vel is not None else None,
            vel_bev=detections.vel_bev if detections.vel_bev is not None else None,
            track_center_history=(
                detections.track_center_history if detections.track_center_history is not None else None
            ),
            track_bev_pos_history=(
                detections.track_bev_pos_history if detections.track_bev_pos_history is not None else None
            ),
        )

    def _infer_color_rgb(self, patch: np.ndarray) -> str:  # noqa: PLR0911
        """Infer color from image patch using RGB color space."""
        if patch.size == 0:
            return ""
        mean_rgb = patch.reshape(-1, 3).mean(axis=0)
        r, g, b = mean_rgb.astype(float)

        if r > 160 and g > 160 and b > 160:
            return "white"
        if r < 75 and g < 75 and b < 75:
            return "black"
        if r > 160 and g < 80 and b < 80:
            return "red"
        if g > 160 and r < 80 and b < 80:
            return "green"
        if b > 160 and r < 80 and g < 80:
            return "blue"
        if r > 160 and g > 160 and b < 80:
            return "yellow"

        return "gray"

    def _infer_color_hsv(self, patch: np.ndarray) -> str:
        """Infer color from image patch using HSV color space."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        # Use mean for simplicity
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        # Achromatic branch
        achro_s_max = 110
        black_v_max = 35
        white_v_min = 120
        gray_v_min = 85

        if mean_s < achro_s_max:
            if mean_v < black_v_max:
                return "black"
            if mean_v >= white_v_min:
                return "white"
            if mean_v >= gray_v_min:
                return "gray"
            return "black"

        # Chromatic: find hue bin
        hue_color_bins = [
            (0, 20, "red"),
            (20, 50, "yellow"),
            (50, 90, "green"),
            (90, 145, "blue"),
            (145, 180, "red"),
        ]
        for h_start, h_end, name in hue_color_bins:
            if h_start <= mean_h < h_end:
                return name
        return "unknown"
