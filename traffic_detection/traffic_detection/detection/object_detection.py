from typing import Literal

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

from traffic_detection.definitions import Box2D
from traffic_detection.detection.base_model import Detector2DBaseModel
from traffic_detection.utils.nms import non_max_suppression

AVAILABLE_HUGGINFACE_MODELS = [
    "ArrayDice/Vehicle_Detection_Model",
    "hilmantm/detr-traffic-accident-detection",
    "SenseTime/deformable-detr",
    "PekingU/rtdetr_v2_r18vd",
    "PekingU/rtdetr_v2_r34vd",
    "PekingU/rtdetr_v2_r50vd",
    "PekingU/rtdetr_v2_r101vd",
    "PekingU/rtdetr_r50vd_coco_o365",
]


class VehicleDetector2DHuggingFace(Detector2DBaseModel):
    """A dummy inference engine for running the model."""

    def __init__(
        self,
        model_id: Literal[
            "ArrayDice/Vehicle_Detection_Model",
            "hilmantm/detr-traffic-accident-detection",
            "SenseTime/deformable-detr",
            "PekingU/rtdetr_v2_r18vd",
            "PekingU/rtdetr_v2_r34vd",
            "PekingU/rtdetr_v2_r50vd",
            "PekingU/rtdetr_v2_r101vd",
            "PekingU/rtdetr_r50vd_coco_o365",
        ] = "PekingU/rtdetr_v2_r50vd",
        device: Literal["cpu", "cuda"] = "cuda",
        confidence_threshold: float = 0.3,
        apply_nms: bool = False,
        nms_class_wise: bool = False,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold, device=device)

        self.image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)

        self.apply_nms = apply_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_class_wise = nms_class_wise

    @torch.no_grad()
    def __call__(self, input_data: np.ndarray) -> Box2D:  # noqa: D102
        input_data = input_data[None, ...]  # add dummy batch dim
        inputs = self.image_processor(images=input_data, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        target_sizes = [[input_data.shape[1], input_data.shape[2]] for b in range(input_data.shape[0])]
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)

        assert len(results) == 1, "Only implemented for batch_size=1"

        pred_boxes = [results[bidx]["boxes"].detach().cpu().numpy() for bidx in range(len(results))][0]
        pred_labels = [results[bidx]["labels"].detach().cpu().numpy() for bidx in range(len(results))][0]
        pred_labels = [self.model.config.id2label[label_idx] for label_idx in pred_labels]
        pred_scores = [np.array(results[bidx]["scores"].detach().cpu().numpy()) for bidx in range(len(results))][0]

        if self.apply_nms:
            keep = non_max_suppression(
                pred_boxes,
                pred_scores,
                pred_labels,
                iou_threshold=self.nms_iou_threshold,
                class_wise=self.nms_class_wise,
            )
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = [pred_labels[i] for i in keep]

        return Box2D(boxes=pred_boxes, scores=pred_scores, labels=np.array(pred_labels))


"""
if __name__ == "__main__":
    from pathlib import Path
    from traffic_detection.utils.image import draw_boxes, draw_keypoints, load_image, save_image

    model = VehicleDetector2DHuggingFace(
        model_id="PekingU/rtdetr_v2_r101vd",
        device="cuda",
        confidence_threshold=0.3,
        apply_nms=True,
    )

    local_images = [
        f"{Path.home()}/dev-playground/traffic_detection/data/output.jpg",
    ]

    for local_img_path in local_images:
        local_img = load_image(local_img_path, max_size=1024)
        boxes2d = model(local_img)
        print("input:   ", local_img.shape, local_img.min(), local_img.max())
        print("output:  ", boxes2d.boxes.shape, boxes2d.boxes.min(), boxes2d.boxes.max())

        image_vis = draw_keypoints(
            draw_boxes(local_img, boxes2d),
            points=boxes2d.boxes_centers,
            points_classes=None,
        )
        output_depth_img = local_img_path.replace(".jpg", "_detect.png")
        save_image(img=image_vis, output_path=output_depth_img, normalize_func="depth")
"""
