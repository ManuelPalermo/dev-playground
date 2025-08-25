import time
from pathlib import Path
from typing import Literal, NamedTuple

import cv2
import numpy as np
import torch

from industrial_detection.definitions import Box2D
from industrial_detection.models.depth_estimation import (
    DEFAULT_DEPTH_MODEL,
    DepthONNXModel,
)
from industrial_detection.models.semantic_segmentation import (
    DEFAULT_SEMSEG_DECODER,
    DEFAULT_SEMSEG_ENCODER,
    INPUT_SHAPE,
    SAM2Image,
    SAM2InputData,
)
from industrial_detection.models.zero_shot_object_detection import (
    ZeroShotObjectDetection2D,
)
from industrial_detection.utils.geometric import calculate_normals_vector
from industrial_detection.utils.image import (
    draw_boxes,
    draw_keypoints,
    draw_masks,
    draw_vectors2d,
    load_image,
    normalize_img,
    resize_img,
    save_image,
)


class PickingNormalsOutput(NamedTuple):
    """Holds Output of PickingNormalsPipeline.

    Attributes:
        depth: The depth map of the image. Shape: [H, W]
        detections: Box2D object containing detection results.
        boxes_centers: Coordinates of the centers of detected bounding boxes. Shape: [N, 2]
        semseg_mask: Semantic segmentation mask of the scene. Shape: [H, W], where each pixel value is the class.
        normals_list3d: List of surface normals in 3D space. Shape: [N, 3]
        normals_list2d: List of surface normals projected to 2D image space. Shape: [N, 2]
        surface_points_mask: Similar to semseg_mask but containing normals surface for each detection. Shape: [H, W]
    """

    depth: np.ndarray
    detections: Box2D
    boxes_centers: np.ndarray
    semseg_mask: np.ndarray
    normals_list3d: np.ndarray
    normals_list2d: np.ndarray
    surface_points_mask: np.ndarray


class PickingNormalsPipeline:
    """Pipeline to detect objects and optimal picking vector from an input image.

    It integrates depth estimation, object detection, instance segmentation, and surface normal calculation
    to provide all necessary outputs for robotic picking tasks.
    """

    def __init__(self, device: Literal["cpu", "cuda"], infer_semseg_on_depth: bool = False) -> None:
        """Initializes the main processing class for industrial detection tasks.

        Args:
            device: The device to use for model inference.
            infer_semseg_on_depth: If True, perform instance segmentation on depth images instead of RGB.
        """
        self.device = device
        self.infer_semseg_on_depth = infer_semseg_on_depth

        self.depth_model = DepthONNXModel(
            model_path=DEFAULT_DEPTH_MODEL,
            inference_providers=("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider",),
        )

        self.detection_prompts = [
            "cardboard box",
            "box package",
            "soft package",
            "white soft package",
            "brown soft package",
            "delivery package",
            "object",
        ]
        # TODO: also use onnx model for less dependencies?
        self.detection_model = ZeroShotObjectDetection2D(
            model_id="google/owlv2-base-patch16-ensemble",
            device=device,
        )

        self.instance_semseg_model = SAM2Image(
            encoder_path=DEFAULT_SEMSEG_ENCODER,
            decoder_path=DEFAULT_SEMSEG_DECODER,
            mask_threshold=0.1,
            point_query_mode="pointlist",
            inference_providers=("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider",),
        )

    def __call__(self, image: np.ndarray) -> PickingNormalsOutput:
        """Run the full picking normals pipeline on a single image."""
        start_t = time.perf_counter()

        # estimate pixels depth
        start_t_depth = time.perf_counter()
        depth = self.estimate_depth(image)
        print("Inference time Depth:", time.perf_counter() - start_t_depth)

        # detect packages
        start_t_det = time.perf_counter()
        detections = self.detect_packages(image)
        boxes_centers = detections.boxes_centers
        print("Inference time Detection:", time.perf_counter() - start_t_det)

        # segmentation for each detection
        start_t_semseg = time.perf_counter()
        semseg_mask = self.instance_segmentation(image, detections)
        print("Inference time Semseg:", time.perf_counter() - start_t_semseg)

        # 4. Surface normals
        start_t_normals = time.perf_counter()
        normals_list3d, normals_list2d, surface_points_mask = self.estimate_surface_normals(
            depth_img=depth,
            rgb_image=image,
            semseg_mask=semseg_mask,
            center_points=boxes_centers,
        )
        print("Inference time Normals:", time.perf_counter() - start_t_normals)

        print("Inference time Pipeline:", time.perf_counter() - start_t)
        return PickingNormalsOutput(
            depth=depth,
            detections=detections,
            boxes_centers=boxes_centers,
            semseg_mask=semseg_mask,
            normals_list3d=normals_list3d,
            normals_list2d=normals_list2d,
            surface_points_mask=surface_points_mask,
        )

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from the input image."""
        # prepare inputs
        image_for_depth = normalize_img(
            resize_img(image, max_size=768),
            normalize_func="uint8_to_float",
        )

        # inference
        depth_raw = self.depth_model(image_for_depth)["depth"]

        # process outputs
        depth = resize_img(
            depth_raw,
            size=(image.shape[1], image.shape[0]),
            interpolation_method=cv2.INTER_NEAREST,
        )
        return depth

    def detect_packages(self, image: np.ndarray) -> Box2D:
        """Detect packages in the input image."""
        # prepare inputs
        image_for_det = resize_img(image, max_size=1024)

        # inference
        detections_raw = self.detection_model.detect(
            image_for_det,
            self.detection_prompts,
            confidence_treshold=0.25,
            apply_nms=True,
            nms_class_wise=False,
        )
        # process outputs
        img_resize_fx = image.shape[1] / image_for_det.shape[1]
        img_resize_fy = image.shape[0] / image_for_det.shape[0]
        detections = Box2D(
            boxes=detections_raw.boxes * [img_resize_fx, img_resize_fy, img_resize_fx, img_resize_fy],
            labels=detections_raw.labels,
            scores=detections_raw.scores,
        )
        return detections

    def instance_segmentation(self, image: np.ndarray, detections: Box2D) -> np.ndarray:
        """Segment pixels of each detected package in the input image."""
        # prepare inputs
        image_for_semseg = resize_img(image, size=INPUT_SHAPE)
        img_resize_fx = image_for_semseg.shape[1] / image.shape[1]
        img_resize_fy = image_for_semseg.shape[0] / image.shape[0]
        boxes_centers_for_semseg = detections.boxes_centers * [img_resize_fx, img_resize_fy]
        boxes_centers_labels = np.array([l for l in range(len(detections.labels))], dtype=np.int32)

        # inference
        semseg_input_data = SAM2InputData(
            img=image_for_semseg,
            query_points=boxes_centers_for_semseg,
            query_labels=boxes_centers_labels,
        )
        semseg_raw = self.instance_semseg_model(input_data=semseg_input_data)

        # process outputs
        semseg_mask = resize_img(
            semseg_raw["masks"].astype(np.float32),
            size=(image.shape[1], image.shape[0]),
            interpolation_method=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        return semseg_mask

    def estimate_surface_normals(
        self, depth_img: np.ndarray, rgb_image: np.ndarray, semseg_mask: np.ndarray, center_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate surface normals from the depth image, RGB image, and segmentation mask."""
        return calculate_normals_vector(
            depth_img=depth_img,
            rgb_image=rgb_image,
            semseg_mask=semseg_mask,
            center_points=center_points,
        )


# NOTE: add img input/output as script args for easier use (argparse or click library)
def main() -> None:  # noqa: D103
    picking_model = PickingNormalsPipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        infer_semseg_on_depth=False,
    )

    local_images = [
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9102.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9103.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9104.jpeg",
    ]

    for local_img_path in local_images:
        image_orig = load_image(local_img_path)

        picking_outputs = picking_model(image_orig)
        box_centers_ids = np.array([l for l in range(len(picking_outputs.detections.labels))], dtype=np.int32)

        # -------------------- visualize all step results ---------------------

        # visualize depth
        save_image(
            img=picking_outputs.depth,
            output_path=local_img_path.replace("/resources/", "/results/").replace(".jpeg", "_depth.png"),
            normalize_func="depth",
        )

        # visualize box detections
        boxes_overlay_img = draw_keypoints(
            image=draw_boxes(image_orig, picking_outputs.detections),
            points=picking_outputs.boxes_centers,
            points_classes=box_centers_ids,
        )
        save_image(
            img=boxes_overlay_img,
            output_path=local_img_path.replace("/resources/", "/results/").replace(".jpeg", "_objects.jpeg"),
        )

        # visualize instance segmentation
        mask_raw_overlay_img = draw_keypoints(
            image=draw_masks(image=image_orig, masks=picking_outputs.semseg_mask),
            points=picking_outputs.boxes_centers,
            points_classes=box_centers_ids,
        )
        output_semseg_img = (
            local_img_path.replace("_depth.png", "_depth_semseg_raw.png")
            if picking_model.infer_semseg_on_depth
            else local_img_path.replace("/resources/", "/results/").replace(".jpeg", "_semseg_raw.png")
        )
        save_image(img=mask_raw_overlay_img, output_path=output_semseg_img, normalize_func="min_max")

        # visualize surface normals
        normals_overlay_img = draw_vectors2d(
            image=draw_keypoints(
                image=draw_masks(
                    image=image_orig,
                    masks=picking_outputs.surface_points_mask,
                ),
                points=picking_outputs.boxes_centers,
                points_classes=box_centers_ids,
            ),
            center_points=picking_outputs.boxes_centers,
            vectors=picking_outputs.normals_list2d,
        )
        save_image(
            img=normals_overlay_img,
            output_path=local_img_path.replace("/resources/", "/results/").replace(".jpeg", "_normals.jpeg"),
            normalize_func="min_max",
        )


if __name__ == "__main__":
    main()
