import pytest

from traffic_detection.detection.detector_factory import (
    ObjectDetectionONNXYOLOV10X,
    VehicleDetector2DHuggingFace,
    ZeroShotObjectDetection2DHuggingFace,
    get_detector_model,
)


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("SenseTime/deformable-detr", VehicleDetector2DHuggingFace),
        ("PekingU/rtdetr_v2_r18vd", VehicleDetector2DHuggingFace),
        ("zero_shot_owlvit_base_patch32", ZeroShotObjectDetection2DHuggingFace),
        ("owlv2_base_patch16_ensemble", ZeroShotObjectDetection2DHuggingFace),
        ("yolov10x_onnx", ObjectDetectionONNXYOLOV10X),
    ],
)
def test_get_detector_model(model_name: str, expected_type: type) -> None:
    """Test that the correct model type is returned for various model detectors."""
    model = get_detector_model(model_name=model_name, device="cpu")
    assert isinstance(model, expected_type)
