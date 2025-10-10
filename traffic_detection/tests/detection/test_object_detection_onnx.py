from pathlib import Path

from traffic_detection.definitions import Box2D
from traffic_detection.detection.object_detection_onnx import ObjectDetectionONNXYOLOV10X
from traffic_detection.utils.image import load_image


def test_object_detection_onnx_model(project_resources_folder: Path, device: str) -> None:
    """Test the ONNX object detection model with a sample image."""
    # GIVEN an ONNX object detection model
    model = ObjectDetectionONNXYOLOV10X(
        confidence_threshold=0.0000001,  # very low threshold to ensure detection
        precision="float32",
        inference_providers=("CPUExecutionProvider",) if device == "cpu" else ("CUDAExecutionProvider",),
    )

    # WHEN the model is called with an image
    local_img_path = project_resources_folder / "output.jpg"
    local_img = load_image(local_img_path)
    boxes2d = model(local_img)

    # THEN the model should return boxes, labels, and scores
    assert isinstance(boxes2d, Box2D), "Model output should be of type Box2D"
    assert boxes2d.boxes.shape[0] > 0, "No boxes detected"
    assert boxes2d.labels.shape[0] > 0, "No labels detected"
    assert boxes2d.scores.shape[0] > 0, "No scores detected"
    assert boxes2d.boxes.shape[1] == 4, "Boxes should have shape (N, 4)"
    assert boxes2d.scores.shape[0] == boxes2d.labels.shape[0], "Scores and labels should match in length"
