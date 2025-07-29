import time
from pathlib import Path

import torch

from industrial_detection.utils.image import draw_boxes, load_image, save_image
from industrial_detection.vlm_object_detection.zero_shot_object_detection import ZeroShotObjectDetection2D


# NOTE: add img input/output as script args for easier use (argparse or click library)
def main() -> None:
    """Main function to run detection and visualization."""
    input_image_path = (
        f"{Path.home()}/dev-playground/industrial_detection/resources/vlm_object_detection/VLM_Scenario-image.jpeg"
    )
    image = load_image(input_image_path)

    detector = ZeroShotObjectDetection2D(
        model_id="google/owlv2-base-patch16-ensemble",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # example 1) concrete classes that we want to detect
    prompts1 = [
        "scissors",
        "pen",
        "duct tape roll",
        "console controller",
        "screw driver",
        "usb port input interface",
    ]

    start1 = time.perf_counter()
    detections1 = detector.detect(image, prompts1, confidence_treshold=0.2)
    print("Time1:", time.perf_counter() - start1)
    save_image(
        img=draw_boxes(image, detections1),
        output_path=input_image_path.replace("/resources/", "/results/").replace(".jpeg", "_objects.jpeg"),
    )

    # example 2) more abstract class definition
    prompts2 = [
        "object which can be picked up",
        "pickable small object",
    ]
    start2 = time.perf_counter()
    detections2 = detector.detect(image, prompts2, confidence_treshold=0.2)
    print("Time2:", time.perf_counter() - start2)
    save_image(
        draw_boxes(image, detections2),
        output_path=input_image_path.replace("/resources/", "/results/").replace(".jpeg", "_abstract.jpeg"),
    )

    # example 3) very low confidence threshold for detections, but apply NMS on top to filter overlapping predictions
    #            NOTE: we could probably also mask out regions of the image which we are not interested:
    #               - draw boxes for zones to ignore, and then use IoU score to filter out 2d boxes in those areas
    #               - or instead define areas of interest and ignore all predictions outside
    prompts3 = [
        "scissors",
        "pen",
        "duct tape roll",
        "console controller",
        "screw driver",
        "usb port input interface",
        "salient objects placed on the table",
        "object which can be picked up",
        "pickable small object",
        "small object placed on top of the table",
    ]
    start3 = time.perf_counter()
    detections3 = detector.detect(image, prompts3, confidence_treshold=0.02)
    print("Time3:", time.perf_counter() - start3)
    save_image(
        draw_boxes(image, detections3),
        output_path=input_image_path.replace("/resources/", "/results/").replace(
            ".jpeg", "_abstract_0.025_no_nms.jpeg"
        ),
    )

    prompts4 = [
        "scissors",
        "pen",
        "duct tape roll",
        "console controller",
        "screw driver",
        "usb port input interface",
        "salient objects placed on the table",
        "object which can be picked up",
        "pickable small object",
        "small object placed on top of the table",
    ]
    start4 = time.perf_counter()
    detections4 = detector.detect(image, prompts4, confidence_treshold=0.2, apply_nms=True, nms_class_wise=False)
    print("Time4:", time.perf_counter() - start4)
    save_image(
        draw_boxes(image, detections4),
        output_path=input_image_path.replace("/resources/", "/results/").replace(".jpeg", "_abstract_0.025_nms.jpeg"),
    )


if __name__ == "__main__":
    main()
