from typing import Any

import numpy as np

from traffic_detection.configs.definitions import (
    EU_DASHED_LANE_LENGTH,
    EU_EMERGENCY_LANE_WIDTH,
    EU_HIGHWAY_DIVIDER_WIDTH,
    EU_LANE_WIDTH,
)


def demo_site_config() -> dict[str, Any]:
    """Demo site configuration for traffic detection."""
    site_config = {
        # ----- polygons areas to count vehicles inside (in pixels) -------
        "count_area_polygons": {
            "coming": np.array([[95, 575], [550, 575], [610, 355], [460, 355]], dtype=np.int32),
            "going": np.array([[690, 575], [1120, 575], [815, 355], [670, 355]], dtype=np.int32),
        },
        # ----- polygon areas with correspondence between pixels and world points for camera to bev transformation ----
        "perspective_area_pixels": np.array(
            [
                [95, 575],
                [1120, 575],
                [815, 355],
                [460, 355],
            ],
            dtype=np.int32,
        ),
        "perspective_area_world": np.array(
            [
                [0, 0],
                [2 * EU_EMERGENCY_LANE_WIDTH + 2 * 2 * EU_LANE_WIDTH + EU_HIGHWAY_DIVIDER_WIDTH, 0],
                [
                    2 * EU_EMERGENCY_LANE_WIDTH + 2 * 2 * EU_LANE_WIDTH + EU_HIGHWAY_DIVIDER_WIDTH,
                    4 * EU_DASHED_LANE_LENGTH,
                ],
                [0, 4 * EU_DASHED_LANE_LENGTH],
            ],
            dtype=np.float32,
        ),
    }
    return site_config
