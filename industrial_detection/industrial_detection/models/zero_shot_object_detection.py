from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from industrial_detection.definitions import Box2D
from industrial_detection.utils.nms import non_max_suppression


class ZeroShotObjectDetection2D:
    """Vision-language model for 2D box detection.

    Attributes:
        processor: Model processor.
        model: Detection model.
    """

    def __init__(
        self,
        model_id: Literal[
            "google/owlv2-base-patch16-ensemble",
            "google/owlv2-large-patch14-ensemble",
            "IDEA-Research/grounding-dino-tiny",
            "IDEA-Research/grounding-dino-base",
        ] = "google/owlv2-base-patch16-ensemble",
        device: Literal["cpu", "cuda"] = "cuda",
    ) -> None:
        """Initializes the processor and model.

        Args:
            model_id: Model identifier.
            device: Device to run the model on.
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        prompts: list[str],
        confidence_treshold: float = 0.3,
        *,
        apply_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        nms_class_wise: bool = True,
    ) -> Box2D:
        """Runs inference and parses 2D boxes.

        Args:
            image: Input image.
            prompts: List of text queries.
            confidence_treshold: Confidence threshold.
            apply_nms: Whether to apply Non-Maximum Suppression.
            nms_iou_threshold: IoU threshold for NMS.
            nms_class_wise: Whether to apply NMS class-wise.

        Returns:
            Box2D: Detection results.
        """
        inputs = self.processor(text=prompts, images=image, return_tensors="pt").to(device=self.device)

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_treshold,
        )[0]
        boxes = results["boxes"].to("cpu").numpy().astype(np.float32)
        scores = results["scores"].to("cpu").numpy().astype(np.float32)

        if isinstance(results["labels"], torch.Tensor):
            labels = [str(prompts[i]) for i in results["labels"]]
        else:
            labels = results["labels"]

        if apply_nms:
            keep = non_max_suppression(
                boxes, scores, labels, iou_threshold=nms_iou_threshold, class_wise=nms_class_wise
            )
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]

        return Box2D(boxes=boxes, scores=scores, labels=np.array(labels))
