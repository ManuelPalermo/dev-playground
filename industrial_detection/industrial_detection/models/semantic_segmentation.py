from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import onnxruntime

from industrial_detection.models.base_model import PerceptionBaseONNXModel
from industrial_detection.utils.image import draw_masks, load_image, resize_img, save_image

"""
original repo: https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases

curl -L https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases/download/0.2.0/decoder.onnx -o sam2decoder.onnx.onnx
curl -L https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases/download/0.2.0/sam2_hiera_base_plus_encoder.onnx -o sam2_hiera_base_plus_encoder.onnx
curl -L https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases/download/0.2.0/sam2_hiera_large_encoder.onnx -o sam2_hiera_large_encoder.onnx
"""

DEFAULT_SEMSEG_DECODER = (
    "https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases/download/0.2.0/decoder.onnx"
)
DEFAULT_SEMSEG_ENCODER = (
    "https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything/releases/download/0.2.0/sam2_hiera_large_encoder.onnx"
)

RGB_CHANNELS = 3

# harcoded values inside the SAM onnx model used
SCALE_FACTOR = 4
INPUT_SHAPE = (1024, 1024)


class SAM2InputData(NamedTuple):  # noqa: D101
    img: np.ndarray
    query_points: list[np.ndarray | tuple[int, int]] | None = None
    query_labels: list[np.ndarray | int] | None = None


class SAM2Image:
    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        mask_threshold: float = 0.0,
        inference_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        inference_options: onnxruntime.SessionOptions | None = None,
        precision: Literal["float32", "float16"] = "float32",
        num_point_queries: int = 25,
        point_query_mode: Literal["center", "center_normal", "random", "grid", "pointlist"] = "center_normal",
    ) -> None:
        self.encoder = SAM2ImageEncoder(
            model_path=encoder_path,
            inference_providers=inference_providers,
            inference_options=inference_options,
            precision=precision,
        )
        self.decoder = SAM2ImageDecoder(
            model_path=decoder_path,
            inference_providers=inference_providers,
            inference_options=inference_options,
            precision=precision,
            mask_threshold=mask_threshold,
        )
        self.num_point_queries = num_point_queries
        self.point_query_mode = point_query_mode

    def __call__(self, input_data: SAM2InputData) -> dict[str, np.ndarray]:
        """Process an input image."""

        input_img_h, input_img_w = input_data.img.shape[0], input_data.img.shape[1]
        encoder_features = self.encoder(input_data.img)

        group_query_points, group_query_labels = self.prepare_input_queries(input_data)
        assert isinstance(group_query_points, list) and isinstance(group_query_labels, list)

        # run decoder for each group of points
        output_masks = np.zeros(shape=(input_img_h, input_img_w), dtype=np.int32)
        output_scores = []
        for input_points, input_label in zip(group_query_points, group_query_labels):
            group_outputs = self.decoder(
                input_data={
                    **encoder_features,
                    "point_coords": input_points,
                    "point_labels": input_label,
                }
            )
            score = group_outputs["score"][0]
            if score > self.decoder.mask_threshold:
                output_mask = resize_img(group_outputs["masks"], size=(input_img_h, input_img_w)) * int(input_label[0])
                output_masks = np.maximum(output_masks, output_mask)
                output_scores.append(score)

        return {"masks": output_masks, "score": np.array(output_scores)}

    def prepare_input_queries(self, input_data: SAM2InputData) -> tuple[list[np.ndarray], list[np.ndarray]]:  # noqa:D102
        if self.point_query_mode == "center":
            # center point of the image
            query_points = [np.array([[INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2]])]
            query_labels = [np.array([1])]

        if self.point_query_mode == "center_normal":
            # random sampling with high bias towards the center of the image
            query_points = [
                [
                    np.random.normal(  # type: ignore
                        loc=(INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2),
                        scale=(INPUT_SHAPE[0] // 5, INPUT_SHAPE[1] // 5),
                        size=(2,),
                    )
                ]
                for _ in range(self.num_point_queries)
            ]
            query_labels = [np.array([i]) for i in range(1, len(query_points) + 1)]

        elif self.point_query_mode == "random":
            query_points = [
                [
                    np.random.randint(  # type: ignore
                        low=(0, 0),
                        high=(INPUT_SHAPE[0] - 1, INPUT_SHAPE[1] - 1),
                        size=(2,),
                    )
                ]
                for _ in range(self.num_point_queries)
            ]
            query_labels = [np.array([i]) for i in range(1, len(query_points) + 1)]

        elif self.point_query_mode == "grid":
            # randomly sample points in a grid pattern
            num_nxn_grid = int(np.sqrt(self.num_point_queries))
            rows = np.linspace(
                0,
                INPUT_SHAPE[0] - 1,
                num_nxn_grid + 2,
                dtype=np.int32,
            )[1:-1]  # ignore first/last px
            cols = np.linspace(
                0,
                INPUT_SHAPE[1] - 1,
                num_nxn_grid + 2,
                dtype=np.int32,
            )[1:-1]  # ignore first/last px
            query_points = [np.array([[x, y]]) for x in rows for y in cols]
            query_labels = [np.array([i]) for i in range(1, len(query_points) + 1)]

        elif self.point_query_mode == "pointlist":
            assert input_data.query_points is not None
            assert input_data.query_labels is not None
            query_points = [np.array([i]) for i in input_data.query_points]
            query_labels = [np.array([i]) for i in input_data.query_labels]

        return query_points, query_labels


class SAM2ImageEncoder(PerceptionBaseONNXModel):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_SEMSEG_ENCODER,
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
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_data = (input_data / 255.0 - mean) / std
        input_data = input_data.transpose(2, 0, 1)
        input_tensor = input_data[np.newaxis, :, :, :].astype(np.float32)

        assert len(self.input_names) == 1
        return {self.input_names[0]: input_tensor}

    def postprocess(self, raw_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: D102
        return raw_output


class SAM2ImageDecoder(PerceptionBaseONNXModel):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_SEMSEG_ENCODER,
        inference_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        inference_options: onnxruntime.SessionOptions | None = None,
        precision: Literal["float32", "float16"] = "float32",
        mask_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            model_path=model_path,
            inference_providers=inference_providers,
            inference_options=inference_options,
            precision=precision,
        )
        self.mask_threshold = mask_threshold

    def preprocess(self, input_data: Any) -> dict[str, np.ndarray]:  # noqa: D102
        input_point_coords, input_point_labels = self._prepare_points(
            np.array(input_data["point_coords"], dtype=np.float32),
            np.array(input_data["point_labels"], dtype=np.float32),
        )

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                INPUT_SHAPE[0] // SCALE_FACTOR,
                INPUT_SHAPE[1] // SCALE_FACTOR,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array([INPUT_SHAPE[0], INPUT_SHAPE[1]], dtype=np.int32)

        input_tensors = {
            "image_embed": input_data["image_embed"],
            "high_res_feats_0": input_data["high_res_feats_0"],
            "high_res_feats_1": input_data["high_res_feats_1"],
            "point_coords": input_point_coords,
            "point_labels": input_point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": original_size,
        }
        return input_tensors

    def postprocess(self, raw_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: D102
        iou_predictions = raw_output["iou_predictions"].squeeze(axis=0)
        masks = raw_output["masks"]
        masks = masks > self.mask_threshold
        masks = masks.astype(np.uint8).squeeze()
        return {"masks": np.array(masks), "score": np.array(iou_predictions)}

    def _prepare_points(self, point_coords: np.ndarray, point_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert point_coords.ndim == 2 and point_labels.ndim == 1, (  # noqa: PLR2004, PT018
            f"Got shapes: {point_coords.shape=} {point_labels.shape=}"
        )
        input_point_coords = point_coords[np.newaxis, ...]
        input_point_labels = point_labels[np.newaxis, ...]
        input_point_coords[..., 0] = input_point_coords[..., 0] / INPUT_SHAPE[1] * INPUT_SHAPE[1]  # Normalize x
        input_point_coords[..., 1] = input_point_coords[..., 1] / INPUT_SHAPE[0] * INPUT_SHAPE[0]  # Normalize y
        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)


if __name__ == "__main__":
    local_images = [
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9102.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9103.jpeg",
        f"{Path.home()}/dev-playground/industrial_detection/resources/picking_normals_angle/IMG_9104.jpeg",
    ]
    infer_on_depth = False
    if infer_on_depth:
        local_images = [img.replace("/resources/", "/outputs/").replace(".jpeg", "_depth.png") for img in local_images]

    model = SAM2Image(
        encoder_path=DEFAULT_SEMSEG_ENCODER,
        decoder_path=DEFAULT_SEMSEG_DECODER,
        mask_threshold=0.90,  # only keep high confidence masks
        num_point_queries=36,
        point_query_mode="center_normal",
    )

    for local_img_path in local_images:
        local_img = load_image(local_img_path, size=INPUT_SHAPE, normalize_func=None)

        output_result = model(input_data=SAM2InputData(img=local_img))
        print("input:   ", local_img.shape, local_img.min(), local_img.max())
        print(
            "output:  ",
            output_result["masks"].shape,
            output_result["masks"].min(),
            output_result["masks"].max(),
            "| Scores: ",
            [round(s, 2) for s in output_result["score"].tolist()],
        )

        mask_overlay_img = draw_masks(image=local_img, masks=output_result["masks"])
        output_semseg_img = (
            local_img_path.replace("_depth.png", "_depth_semseg.png")
            if infer_on_depth
            else local_img_path.replace("/resources/", "/outputs/").replace(".jpeg", "_semseg.png")
        )
        save_image(img=mask_overlay_img, output_path=output_semseg_img, normalize_func="min_max")
