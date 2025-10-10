import cv2
import numpy as np


class CameraToBevTransformation:
    """Transforms points and images from camera view to Bird's Eye View (BEV).

    source: 4 pts in image pixels (u,v)
    target: 4 corresponding points on the ground plane in meters (x,y)
    resolution_m_per_px: desired BEV resolution (meters per BEV pixel)
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, resolution_m_per_px: float = 0.05) -> None:
        self.source = np.array(source, dtype=np.float32)  # (4,2) pixels
        self.target_m = np.array(target, dtype=np.float32)  # (4,2) meters
        self.resolution_m_per_px = float(resolution_m_per_px)

        self.H = cv2.getPerspectiveTransform(self.source, self.target_m)

        # Build BEV mapping and output size
        self._update_bev_mapping()

    def _update_bev_mapping(self) -> None:
        # Bounds of the target quad in meters
        min_xy = self.target_m.min(axis=0)
        max_xy = self.target_m.max(axis=0)
        width_m = float(max_xy[0] - min_xy[0])
        height_m = float(max_xy[1] - min_xy[1])

        # Desired BEV size in pixels
        self.bev_width_px = int(np.ceil(width_m / self.resolution_m_per_px))
        self.bev_height_px = int(np.ceil(height_m / self.resolution_m_per_px))

        # Ground (meters) -> BEV pixels (x right, y up)
        sx = 1.0 / self.resolution_m_per_px
        sy = -1.0 / self.resolution_m_per_px
        tx = -min_xy[0] * sx
        ty = max_xy[1] * (-sy)
        T_g2bev = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Final mapping: Image -> BEV pixels
        self.H_i2bev = T_g2bev @ self.H

    def transform_points(self, points_uv: np.ndarray) -> np.ndarray:
        """Image pixels -> Ground meters."""
        if points_uv.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        reshaped_points = points_uv.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.H)
        return transformed_points.reshape(-1, 2)

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """Warp to BEV canvas at the configured resolution."""
        return cv2.warpPerspective(
            image,
            self.H_i2bev,
            (self.bev_width_px, self.bev_height_px),
        )
