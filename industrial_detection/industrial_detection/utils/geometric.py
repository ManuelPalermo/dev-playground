from typing import Literal

import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor


def get_img_neighbor_points_mask(image_shape: np.ndarray, center_point: np.ndarray, radius: int = 25) -> np.ndarray:
    """Returns the coordinates of neighboring points within a square patch around a center point in an image.

    Args:
        image_shape: Shape of the image as (height, width).
        center_point: The (x, y) coordinates of the center point.
        radius: The radius of the square patch to consider around the center point.

    Returns:
        numpy.ndarray: The mask with neighbor points in image space. Shape: [height, width]
    """
    height, width = image_shape
    point2d_x, point2d_y = center_point

    x_min = max(0, int(point2d_x) - radius)
    x_max = min(width, int(point2d_x) + radius + 1)
    y_min = max(0, int(point2d_y) - radius)
    y_max = min(height, int(point2d_y) + radius + 1)

    mask = np.zeros((height, width), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def calculate_normals_vector(
    center_points: np.ndarray,
    depth_img: np.ndarray,
    rgb_image: np.ndarray,
    semseg_mask: np.ndarray | None = None,
    algorithm: Literal["ransac", "normals_avg"] = "ransac",
    avg_radius: int = 100,
    visualize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates surface normal vectors for given center points in an image using depth and optional semseg data.

    For each center point, a local neighborhood is selected, and the surface normal is estimated
    using either RANSAC plane fitting or a weighted average of local normals.

    Optionally, segmentation masks can be used to restrict the neighborhood to the same class as the center point.

    Args:
        center_points: Array of shape (N, 2) with (x, y) coordinates of center points in image space.
        depth_img: 2D array representing the depth image.
        rgb_image: 3D array representing the RGB image.
        semseg_mask: 2D array with semantic segmentation labels. If provided, neighborhoods are restricted
            to the same class as the center point.
        avg_radius: Radius (in pixels) for the local neighborhood around each center point.
        algorithm: Algorithm to use for normal estimation.
            - "ransac" fits a plane using RANSAC;
            - "normals_avg" computes a weighted average of local normals.
        visualize: if 3d pointcloud and normals should be visualized (blocking window).

    Returns:
        - normals_list3d: Array of shape (N, 3) with estimated 3D normal vectors for each center point.
        - normals_list2d: Array of shape (N, 2) with projected 2D normal vectors for each center point.
        - surface_points_mask: 2D array where each pixel is labeled with the index of the center point it belongs to.
    """
    # Create Open3D depth image + rgb (optional, for visualization)
    depth_img = np.clip(depth_img.astype(np.float32), min=0.01, max=None)
    o3d_depth = o3d.geometry.Image(depth_img)
    o3d_color = o3d.geometry.Image((rgb_image[..., :3]).astype(np.uint8))

    # camera intrinsics (fx, fy, cx, cy, s) - assuming a default pinhole (could be taken from specific camera model)
    height, width = depth_img.shape
    fx = 0.5 * width
    fy = 0.5 * height
    cx = width / 2
    cy = height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Create RGBD pointcloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # For each object instance in segmentation, estimate normals
    surface_points_mask = np.zeros((height, width), dtype=np.float32)
    normals_list3d = []
    normals_list2d = []
    for i, center_point in enumerate(center_points):
        center_point3d_mask = get_img_neighbor_points_mask(
            image_shape=depth_img.shape,
            center_point=center_point,
            radius=0,
        )
        neighbor_img_mask = get_img_neighbor_points_mask(
            image_shape=depth_img.shape,
            center_point=center_point,
            radius=avg_radius,
        )

        if semseg_mask is not None:
            center_point_class = semseg_mask[int(center_point[1]), int(center_point[0])]
            neighbor_img_mask = neighbor_img_mask & (semseg_mask == center_point_class)

        mask_points = np.asarray(pcd.points)[neighbor_img_mask.reshape(-1)]
        center_point_3d = np.asarray(pcd.points)[center_point3d_mask.reshape(-1)]

        if algorithm == "ransac":
            # options 1) use RANSAC plane fit to get normal vector
            # Fit plane: ax + by + cz + d = 0, so fit z = -(a*x + b*y + d)/c
            x = mask_points[:, :2]
            y = mask_points[:, 2]
            ransac = RANSACRegressor(
                min_samples=3,
                residual_threshold=0.01,
                max_trials=200,
                stop_probability=0.99,
            )
            ransac.fit(x, y)  # Z = aX + bY
            a, b = ransac.estimator_.coef_  # coefficients
            normal_3d_vector = np.array([a, b, -1.0])
            normal_3d = normal_3d_vector / np.linalg.norm(normal_3d_vector)
            # TODO: fix select only the points from the neighbor_img_mask which are inliers of the RANSAC optimization
            # inlier_mask = ransac.inlier_mask_
            # neighbor_img_mask = neighbor_img_mask[inlier_mask]

        elif algorithm == "normals_avg":
            # option 2) do a weighted avg of surface normals based on distance to center
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(mask_points)
            # Estimate normals
            obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
            normals = np.asarray(obj_pcd.normals)

            # distances from each point to the center and use that to compute weighted avg (closer have more weight)
            distances = np.linalg.norm(mask_points - center_point_3d, axis=1)
            weights = 1.0 / (distances + 1e-6)
            normal_3d = np.average(normals, axis=0, weights=weights)
            normal_3d = normal_3d / np.linalg.norm(normal_3d)
        else:
            raise NotImplementedError

        # Ensure normal faces the camera
        vector_to_camera = -center_point_3d.squeeze()  # camera at (0,0,0)
        if np.dot(normal_3d, vector_to_camera) > 0:
            normal_3d = -normal_3d

        # project back the 3d normal vectors to 2d camera space for easier visualization
        # TODO: fix, the vectors in 2D dont look as expected for some reason?
        normal_2d = (
            fx * normal_3d[0] / normal_3d[2] + cx,
            fy * normal_3d[1] / normal_3d[2] + cy,
        )
        normal_2d = np.array(normal_2d) - center_point
        normal_2d = normal_2d / np.linalg.norm(normal_2d)

        normals_list3d.append(normal_3d)
        normals_list2d.append(normal_2d)
        surface_points_mask = surface_points_mask + (neighbor_img_mask.astype(np.float32) * i)

    # Visualize 3D point cloud and normals using Open3D
    # cleanup
    if visualize and len(normals_list3d) > 0:
        lines = []
        colors = []
        points = []
        normal_length = max(
            0.0005,
            np.linalg.norm(
                np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()),
            )
            * 0.0005,
        )
        for i, center_point in enumerate(center_points):
            # Project 2D center point to 3D
            mask = get_img_neighbor_points_mask(depth_img.shape, center_point, radius=1).reshape(-1)
            points_3d = np.asarray(pcd.points)
            if points_3d.shape[0] < mask.shape[0]:
                pad_len = mask.shape[0] - points_3d.shape[0]
                points_3d = np.pad(points_3d, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
            center_3d = points_3d[mask].mean(axis=0)
            points.append(center_3d)
            points.append(center_3d + normals_list3d[i] * normal_length)
            lines.append([2 * i, 2 * i + 1])
            colors.append([1, 0, 0])  # red normals

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # Increase line width for better visibility
        pcd_visu = pcd.voxel_down_sample(voxel_size=0.00001)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_visu, line_set])

    return np.array(normals_list3d), np.array(normals_list2d), surface_points_mask
