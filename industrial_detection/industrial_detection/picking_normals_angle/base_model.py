import abc
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import onnxconverter_common
import onnxruntime

DEFAULT_MODEL_CACHE_PATH = (
    f"{Path.home()}/dev-playground/industrial_detection/industrial_detection/picking_normals_angle/.cache_model/"
)


class PerceptionBaseModel(abc.ABC):
    """Inference engine for generic models."""

    def __call__(self, input_data: np.ndarray) -> dict[str, np.ndarray]:
        """Process an input image."""
        input_tensor = self.preprocess(input_data)
        raw_output = self.infer(input_tensor)
        output = self.postprocess(raw_output)
        return output

    @abc.abstractmethod
    def infer(self, input_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Runs inference."""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, input_data: Any) -> dict[str, np.ndarray]:
        """Pre-processes data."""
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, raw_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Post-processes data."""
        raise NotImplementedError


class PerceptionBaseONNXModel(PerceptionBaseModel):
    """Inference for ONNX models."""

    def __init__(
        self,
        model_path: str | Path,
        inference_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        inference_options: onnxruntime.SessionOptions | None = None,
        precision: Literal["float32", "float16"] = "float32",
    ) -> None:
        self.model_path = model_path
        self.inference_providers = inference_providers
        self.inference_options = inference_options
        self.precision = precision

        self.onnxruntime_session = self.load_onnx_model(Path(model_path))
        self.input_names = self.get_input_details()
        self.output_names = self.get_output_details()

    def load_onnx_model(self, model_path: Path) -> onnxruntime.InferenceSession:
        """Loads the onnx model as an onnxruntime session."""

        model_name = model_path.name
        Path(DEFAULT_MODEL_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        target_model_path = Path(DEFAULT_MODEL_CACHE_PATH) / model_name

        if "http" in str(model_path) and not target_model_path.exists():
            # download to local model cache
            subprocess.call(["curl", "-L", str(model_path), "-o", str(target_model_path)])
            assert target_model_path.exists() and ".onnx" in str(target_model_path), (
                f"Valid model ({model_path}) could not be downloaded to: {target_model_path}"
            )

        # NOTE: debug float16, not working properly
        if self.precision == "float16":
            model = onnx.load(target_model_path)
            model_fp16 = onnxconverter_common.float16.convert_float_to_float16(model)
            target_model_path_fp16 = str(target_model_path).replace(".onnx", "_fp16.onnx")
            onnx.save(model_fp16, target_model_path_fp16)
            target_model_path = Path(target_model_path_fp16)

        return onnxruntime.InferenceSession(
            target_model_path,
            providers=self.inference_providers,
            sess_options=self.inference_options,
        )

    def get_input_details(self) -> list[str]:
        """Retrieves the names of the input nodes for the ONNX model."""
        model_inputs = self.onnxruntime_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        return input_names

    def get_output_details(self) -> list[str]:
        """Retrieves the names of the output nodes for the ONNX model."""
        model_outputs = self.onnxruntime_session.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        return output_names

    def __call__(self, input_data: Any) -> dict[str, np.ndarray]:
        """Process an input image."""
        input_tensor = self.preprocess(input_data)

        if self.precision == "float16":
            input_tensor = {k: v.astype(np.float16) for k, v in input_tensor.items()}

        raw_output = self.infer(input_tensor)
        output = self.postprocess(raw_output)
        return output

    def infer(self, input_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Runs inference."""
        outputs = self.onnxruntime_session.run(input_feed=input_tensors, output_names=self.output_names)
        return {name: outputs[idx] for idx, name in enumerate(self.output_names)}

    @abc.abstractmethod
    def preprocess(self, input_data: Any) -> dict[str, np.ndarray]:  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self, raw_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: D102
        raise NotImplementedError
