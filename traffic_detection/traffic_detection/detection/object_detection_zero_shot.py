from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from traffic_detection.definitions import Box2D
from traffic_detection.detection.base_model import Detector2DBaseModel
from traffic_detection.utils.nms import non_max_suppression


class ZeroShotObjectDetection2DHuggingFace(Detector2DBaseModel):
    """Vision-language models for 2D box detection from huggingface.

    Mostly for for offline debugging purposes and comparing models.
    """

    def __init__(
        self,
        prompts: list[str],
        model_id: Literal[
            "google/owlvit-base-patch32",  # runs almost real-time but worse performance
            "google/owlv2-base-patch16-ensemble",  # good detections but slow
            "IDEA-Research/grounding-dino-tiny",  # ok detections but slow
            "IDEA-Research/grounding-dino-base",  # good detections but very slow
        ] = "google/owlvit-base-patch32",
        device: Literal["cpu", "cuda"] = "cuda",
        confidence_threshold: float = 0.3,
        apply_nms: bool = False,
        nms_class_wise: bool = False,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        """Initializes the processor and model.

        Args:
            prompts: List of text queries.
            model_id: Model identifier.
            device: Device to run the model on.
            confidence_threshold: Confidence threshold.
            apply_nms: Whether to apply Non-Maximum Suppression.
            nms_class_wise: Whether to apply NMS class-wise.
            nms_iou_threshold: IoU threshold for NMS.
        """
        super().__init__(confidence_threshold=confidence_threshold, device=device)
        self.prompts = prompts
        self.apply_nms = apply_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_class_wise = nms_class_wise

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    @torch.no_grad()
    def __call__(self, input_data: np.ndarray) -> Box2D:
        """Runs inference and parses 2D boxes."""
        inputs = self.processor(text=self.prompts, images=input_data, return_tensors="pt").to(device=self.device)

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([input_data.shape[:2]])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]
        boxes = results["boxes"].to("cpu").numpy().astype(np.float32)
        scores = results["scores"].to("cpu").numpy().astype(np.float32)

        if isinstance(results["labels"], torch.Tensor):
            labels = [str(self.prompts[i]) for i in results["labels"]]
        else:
            labels = results["labels"]

        Box2D(boxes=boxes, scores=scores, labels=np.array(labels))

        if self.apply_nms:
            keep = non_max_suppression(
                boxes, scores, labels, iou_threshold=self.nms_iou_threshold, class_wise=self.nms_class_wise
            )
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]

        return Box2D(boxes=boxes, scores=scores, labels=np.array(labels))


"""
if __name__ == "__main__":
    from pathlib import Path
    from traffic_detection.utils.image import draw_boxes, draw_keypoints, load_image, save_image

    prompts = ["vehicle", "car", "van", "truck", "motorbike"]
    model = ZeroShotObjectDetection2DHuggingFace(
        prompts=prompts,
        model_id="google/owlv2-base-patch16-ensemble",
        device="cuda",
        confidence_threshold=0.3,
        apply_nms=False,
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
