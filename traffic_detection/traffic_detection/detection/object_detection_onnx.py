from collections.abc import Sequence
from typing import ClassVar, Literal

import numpy as np
import onnxruntime

from traffic_detection.definitions import Box2D
from traffic_detection.detection.base_model import Detector2DBaseModelONNX
from traffic_detection.utils.image import normalize_img, resize_img
from traffic_detection.utils.nms import non_max_suppression

RGB_CHANNELS = 3


class ObjectDetectionONNXYOLOV10X(Detector2DBaseModelONNX):
    """ONNX model for object detection.

    source: https://huggingface.co/onnx-community/yolov10x
    Model signature:
        inputs:
            - name: images  | tensor: float32[1,3,640,640]
        outputs:
            - name: output0 | tensor: float32[1,6,300]
    """

    DETECTION_MODEL = "https://huggingface.co/onnx-community/yolov10x/resolve/main/onnx/model.onnx"
    # relevant classes from the label set
    # https://huggingface.co/onnx-community/yolov10x/blob/main/config.json
    ID2LABEL: ClassVar = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
    }

    def __init__(
        self,
        inference_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        inference_options: onnxruntime.SessionOptions | None = None,
        precision: Literal["float32", "float16"] = "float32",
        confidence_threshold: float = 0.3,
        apply_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        nms_class_wise: bool = True,
    ) -> None:
        super().__init__(
            model_path=self.DETECTION_MODEL,
            inference_providers=inference_providers,
            inference_options=inference_options,
            precision=precision,
        )

        self.confidence_threshold = confidence_threshold
        self.apply_nms = apply_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_class_wise = nms_class_wise

    def __call__(self, input_data: np.ndarray) -> Box2D:  # noqa: D102
        # pre-process
        model_input_shape = (640, 640)
        input_tensor = normalize_img(resize_img(input_data, size=model_input_shape), normalize_func="uint8_to_float")
        input_tensor = np.expand_dims(input_tensor.transpose(2, 0, 1), axis=0).astype(np.float32)
        assert len(self.input_names) == 1

        # onnx inference
        if self.precision == "float16":
            input_tensor = input_tensor.astype(np.float16)

        raw_output = self.infer({self.input_names[0]: input_tensor})

        # post-process
        preds = raw_output["output0"][0, ...]  # (1, 6, 300) -> (300, 6)[x1, y1, x2, y2, score]
        boxes = preds[:, :4]
        scores = preds[:, 4]
        labels_idx = preds[:, 5]

        # Filter by confidence threshold and class labels
        mask_confidence = scores > self.confidence_threshold
        mask_labels = np.isin(labels_idx, list(self.ID2LABEL.keys()))
        mask = mask_confidence & mask_labels
        boxes = boxes[mask]
        scores = scores[mask]
        labels_idx = labels_idx[mask]
        labels = [(self.ID2LABEL[int(lidx)] if int(lidx) in self.ID2LABEL else "") for lidx in labels_idx.tolist()]

        if self.apply_nms:
            keep = non_max_suppression(
                boxes,
                scores,
                labels,
                iou_threshold=self.nms_iou_threshold,
                class_wise=self.nms_class_wise,
            )
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]

        # resize predictions to original img size
        img_resize_fx = input_data.shape[1] / model_input_shape[1]
        img_resize_fy = input_data.shape[0] / model_input_shape[0]
        return Box2D(
            boxes=boxes * [img_resize_fx, img_resize_fy, img_resize_fx, img_resize_fy],
            labels=np.array(labels, dtype=np.str_),
            scores=scores,
        )


"""
if __name__ == "__main__":
    from pathlib import Path
    from traffic_detection.utils.image import draw_boxes, draw_keypoints, load_image, save_image

    device = "cuda"
    model = ObjectDetectionONNXYOLOV10X(
        precision="float32",
        inference_providers=("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider",),
    )

    local_images = [
        f"{Path.home()}/dev-playground/traffic_detection/data/output.jpg",
    ]

    for local_img_path in local_images:
        local_img = load_image(local_img_path)
        boxes2d = model(local_img)
        print("input:   ", local_img.shape, local_img.min(), local_img.max())
        print("output:  ", boxes2d.boxes.shape, boxes2d.boxes.min(), boxes2d.boxes.max(), boxes2d.scores)

        image_vis = draw_keypoints(
            draw_boxes(local_img, boxes2d),
            points=boxes2d.boxes_centers,
        )
        output_depth_img = local_img_path.replace(".jpg", "_detect.png")
        save_image(img=image_vis, output_path=output_depth_img, normalize_func="depth")
"""
