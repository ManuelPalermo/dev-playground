from typing import Literal

from traffic_detection.detection.base_model import Detector2DBaseModel
from traffic_detection.detection.object_detection import AVAILABLE_HUGGINFACE_MODELS, VehicleDetector2DHuggingFace
from traffic_detection.detection.object_detection_onnx import ObjectDetectionONNXYOLOV10X
from traffic_detection.detection.object_detection_zero_shot import ZeroShotObjectDetection2DHuggingFace

VALID_DETECTOR_MODELS = [
    *AVAILABLE_HUGGINFACE_MODELS,
    "zero_shot_owlvit_base_patch32",
    "owlv2_base_patch16_ensemble",
    "yolov10x_onnx",
]


def get_detector_model(
    model_name: Literal[
        "ArrayDice/Vehicle_Detection_Model",
        "hilmantm/detr-traffic-accident-detection",
        "SenseTime/deformable-detr",
        "PekingU/rtdetr_v2_r18vd",
        "PekingU/rtdetr_v2_r34vd",
        "PekingU/rtdetr_v2_r50vd",
        "PekingU/rtdetr_v2_r101vd",
        "PekingU/rtdetr_r50vd_coco_o365",
        "zero_shot_owlvit_base_patch32",
        "owlv2_base_patch16_ensemble",
        "yolov10x_onnx",
    ] = "owlv2_base_patch16_ensemble",
    device: Literal["cpu", "cuda"] = "cpu",
    apply_nms: bool = True,
) -> Detector2DBaseModel:
    """Returns a detector model based on the specified model name."""

    if model_name == "yolov10x_onnx":
        return ObjectDetectionONNXYOLOV10X(
            precision="float32",
            inference_providers=("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider",),
            confidence_threshold=0.3,
            apply_nms=apply_nms,
        )

    if model_name == "zero_shot_owlvit_base_patch32":
        prompts = ["vehicle", "car", "van", "truck", "motorbike"]
        return ZeroShotObjectDetection2DHuggingFace(
            model_id="google/owlvit-base-patch32",
            device=device,
            prompts=prompts,
            confidence_threshold=0.02,  # strangely small threshold?
            apply_nms=apply_nms,
        )

    if model_name == "owlv2_base_patch16_ensemble":
        prompts = ["car", "van", "truck", "bus", "motorbike", "pedestrian"]
        return ZeroShotObjectDetection2DHuggingFace(
            model_id="google/owlv2-base-patch16-ensemble",
            device=device,
            prompts=prompts,
            confidence_threshold=0.3,
            apply_nms=apply_nms,
        )

    if model_name in AVAILABLE_HUGGINFACE_MODELS:
        return VehicleDetector2DHuggingFace(
            model_id=model_name,
            device=device,
            confidence_threshold=0.3,
            apply_nms=apply_nms,
        )

    raise ValueError(f"Unsupported model type: {model_name}. Pick one of: {VALID_DETECTOR_MODELS}")
