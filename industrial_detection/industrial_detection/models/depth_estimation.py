from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime

from industrial_detection.models.base_model import PerceptionBaseONNXModel
from industrial_detection.utils.image import load_image, save_image

"""
original repo: https://github.com/fabio-sim/Depth-Anything-ONNX/releases/

RELEASE=v1.0.0
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vits14.onnx -o depth_anything_vits14.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vitb14.onnx -o depth_anything_vitb14.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_vitl14.onnx -o depth_anything_vitl14.onnx

RELEASE=v2.0.0
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_v2_vits_indoor_dynamic.onnx -o depth_anything_v2_vits_indoor_dynamic.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_v2_vitb_indoor_dynamic.onnx -o depth_anything_v2_vitb_indoor_dynamic.onnx
curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/${RELEASE}/depth_anything_v2_vitl_indoor_dynamic.onnx -o depth_anything_v2_vitl_indoor_dynamic.onnx
"""

# DEFAULT_DEPTH_MODEL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vitl14.onnx"
DEFAULT_DEPTH_MODEL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitl_indoor_dynamic.onnx"

RGB_CHANNELS = 3


class DepthONNXModel(PerceptionBaseONNXModel):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_DEPTH_MODEL,
        inference_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        inference_options: onnxruntime.SessionOptions | None = None,
        precision: Literal["float32", "float16"] = "float32",
    ) -> None:
        super().__init__(
            model_path=model_path,
            inference_providers=inference_providers,
            inference_options=inference_options,
            precision=precision,
        )

    def preprocess(self, input_data: np.ndarray) -> dict[str, np.ndarray]:  # noqa: D102
        if input_data.shape[-1] == RGB_CHANNELS:
            input_tensor = input_data.transpose(2, 0, 1)
        elif input_data.shape[0] == RGB_CHANNELS:
            input_tensor = input_data
        else:
            raise ValueError(f"Invalid input data format: {input_data.shape}")

        # add dummy batch dim | onnx depth models expect [B, C, H, W] input with C=3
        assert len(self.input_names) == 1
        return {self.input_names[0]: input_tensor[None, ...]}

    def postprocess(self, raw_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: D102
        if "v1.0.0" in str(self.model_path):
            output = {k: v[0, ...].transpose(1, 2, 0) for k, v in raw_output.items()}
        else:
            output = {k: v[0, ...][..., None] for k, v in raw_output.items()}

            # NOTE: for some reason the indoor depth model likes to predict foreground objects
            #       as being further away than background :(
            # is predicting inverted Invert depth values (assuming depth is distance from camera, invert so that closer points have higher values)
            for k, v in output.items():
                min_val, max_val = v.min(), v.max()
                inv_depth_img = 1.0 / (v + 1e-6)
                inv_depth_img = (inv_depth_img - inv_depth_img.min()) / (inv_depth_img.max() - inv_depth_img.min())
                inv_depth_img = inv_depth_img * (max_val - min_val) + min_val  # scale back to original range
                output[k] = inv_depth_img

        return output  # [H, W, 1]


if __name__ == "__main__":
    model = DepthONNXModel(model_path=DEFAULT_DEPTH_MODEL, precision="float32")

    local_images = [
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9102.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9103.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9104.jpeg",
    ]

    for local_img_path in local_images:
        local_img = load_image(local_img_path, normalize_func="uint8_to_float", max_size=768)
        output_result = model(local_img)["depth"]
        print("input:   ", local_img.shape, local_img.min(), local_img.max())
        print("output:  ", output_result.shape, output_result.min(), output_result.max())

        output_depth_img = local_img_path.replace("/resources/", "/outputs/").replace(".jpeg", "_depth.png")
        save_image(img=output_result, output_path=output_depth_img, normalize_func="depth")

    # TODO: convert to pointcloud using intrinsics/extrinsics
    # TODO: depth calibration to real depth
