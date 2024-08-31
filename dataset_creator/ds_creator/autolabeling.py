import abc
import json
import os
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from ds_creator.semantic_dataset_creator import load_image
from ds_creator.utils import glob_images


class AutoLabellerModel(abc.ABC):
    NAME = ""

    def __init__(self, model: Any, image_shape: tuple[int, int] | None = None):
        self.model = model
        self.image_shape = image_shape

    def autolabel_directory(self, directory: str, save_path: str | None):
        print(f"\n>>  Running '{self.NAME}.autolabel_directory'")
        for file_path in tqdm(glob_images(directory), desc="Autolabeling"):
            self.label_data(file_path, save_path)

    @torch.inference_mode()
    def label_data(self, filepath: str, save_path: str | None):
        assert self.model is not None
        image = load_image(filepath, shape=self.image_shape)
        outputs = self.predict(image)
        self.process_and_save_labels(outputs, filepath, save_path)

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> Any:
        pass

    @abc.abstractmethod
    def process_and_save_labels(self, outputs: Any, filepath: str, save_path: str | None) -> None:
        pass


class AutoLabellerGroundingSAM(AutoLabellerModel):
    NAME = "GroundingSAM"

    def __init__(self, grounding_sam_class_onthology, image_shape: tuple[int, int] | None = (256, 256)):
        model = GroundedSAM(
            ontology=CaptionOntology(grounding_sam_class_onthology),
            box_threshold=0.35,
            text_threshold=0.25,
        )
        super().__init__(model, image_shape)

    def predict(self, image: np.ndarray) -> Any:
        predictions = self.model.predict(image)

        # filter overlapping boxes
        predictions = predictions.with_nms(threshold=0.9, class_agnostic=True)  # filter different class boxes
        predictions = predictions.with_nms(threshold=0.7, class_agnostic=False)  # filter same class boxes
        return predictions

    def process_and_save_labels(self, outputs: sv.Detections, filepath: str, save_path: str | None) -> None:
        orig_image = load_image(filepath, shape=None)
        orig_image_shape_xy = np.shape(orig_image)[:2][::-1]
        if self.image_shape is None:
            resize_factor = None
        else:
            resize_factor = [
                orig_image_shape_xy[0] / self.image_shape[0],
                orig_image_shape_xy[1] / self.image_shape[1],
            ]

        # parse bbox 2d
        box_labels: dict[str, Any] = {
            "original_filepath": filepath,
            "model": f"{self.model}",
            "onthology": self.model.ontology.promptMap,
            "box_2d": {},
        }

        if outputs is not None:
            for idx, box_2d in enumerate(outputs.xyxy):
                confidence = outputs.confidence[idx] if outputs.confidence is not None else -1
                if outputs.class_id is not None:
                    class_id = outputs.class_id[idx]
                    class_name = self.model.ontology.classes()[class_id]
                else:
                    class_id = -1
                    class_name = ""

                if resize_factor is not None:
                    box_2d = box_2d * [resize_factor[0], resize_factor[1], resize_factor[0], resize_factor[1]]

                box_labels["box_2d"][f"box_{idx}"] = {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "confidence": float(confidence),
                    "box": box_2d.tolist(),
                }

        # parse semseg mask
        semseg_mask = None
        if (outputs is not None) and (outputs.mask is not None and outputs.mask.size > 0):
            # convert bool masks to [0,255] uint8 and resize to original img shape
            semseg_mask = np.transpose(outputs.mask.astype(np.uint8) * 255, axes=(1, 2, 0))

            if resize_factor is not None:
                semseg_mask = cv2.resize(semseg_mask, orig_image_shape_xy, interpolation=cv2.INTER_NEAREST)

            if outputs.mask.shape[0] == 1:
                semseg_mask = semseg_mask[..., None]
            semseg_mask = [semseg_mask[..., idx] for idx in range(semseg_mask.shape[-1])]

        # save label data to disc
        self.save_labels(box_labels, semseg_mask, filepath, save_path)
        self.visualize_predictions(orig_image, outputs, semseg_mask, filepath, save_path)

    def save_labels(self, box_labels, semseg_mask, filepath: str, save_path: str | None) -> None:
        # save path bbox
        dst_file_bbox = (
            filepath.replace(".png", "_box2d.json")
            if save_path is None
            else os.path.join(save_path, os.path.basename(filepath).replace(".png", "_box2d.json"))
        )

        # save semseg mask
        if semseg_mask is not None:
            for idx, mask in enumerate(semseg_mask):
                class_name = box_labels["box_2d"][f"box_{idx}"]["class_name"]
                semseg_dst_file = dst_file_bbox.replace("_box2d.json", f"_semseg_mask_{idx}_{class_name}.png")
                cv2.imwrite(semseg_dst_file, mask)
                box_labels["box_2d"][f"box_{idx}"]["mask_path"] = semseg_dst_file
            tqdm.write(f"Saved SemSeg labels to: {dst_file_bbox.replace('_box2d.json', '_semseg_*.png')}")

        # save bbox
        with open(dst_file_bbox, "w", encoding="utf-8") as outfile_handle:
            os.makedirs(os.path.dirname(dst_file_bbox), exist_ok=True)
            json.dump(box_labels, outfile_handle, indent=4)
        tqdm.write(f"Saved BBox2d labels to: {dst_file_bbox}")

    def visualize_predictions(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        semseg_mask: np.ndarray | None,
        filepath: str,
        save_path: str | None,
    ) -> None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dst_file_annotations = (
            filepath.replace(".png", "_instseg.png")
            if save_path is None
            else os.path.join(save_path, os.path.basename(filepath).replace(".png", "_instseg.png"))
        )

        # draw boxes to scene
        box_annotator = sv.BoxAnnotator(thickness=1, text_scale=0.2, text_padding=2)
        labels = [
            f"{self.model.ontology.classes()[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # draw masks to scene
        if semseg_mask:
            height, width, _ = image.shape
            colors = sv.ColorPalette.DEFAULT

            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for idx, mask in enumerate(semseg_mask):
                color = colors.by_idx(idx)  # type:ignore
                color = np.array([color.r, color.g, color.b]) / 255.0
                idx_color_mask = (
                    ((np.tile(np.squeeze(mask)[..., None], reps=(1, 1, 3)) / 255.0) * color) * 255
                ).astype(np.uint8)
                color_mask = np.maximum(color_mask, idx_color_mask)

            height, width, _ = image.shape
            annotated_frame = sv.draw_image(
                scene=annotated_frame,
                image=color_mask,
                opacity=0.3,
                rect=sv.Rect(x=0, y=0, width=width, height=height),
            )
            dst_file_semseg_visu = dst_file_annotations.replace("_instseg.png", "_semseg.png")
            cv2.imwrite(dst_file_semseg_visu, color_mask)
            tqdm.write(f"Saved SemSeg visu to:  {dst_file_semseg_visu}")

        cv2.imwrite(dst_file_annotations, annotated_frame)
        tqdm.write(f"Saved InstSeg visu to: {dst_file_annotations}")


class AutoLabellerDepthAnything(AutoLabellerModel):
    NAME = "DepthAnything"

    def __init__(
        self,
        pretrained_config: str = "LiheYoung/depth-anything-large-hf",
        image_shape: tuple[int, int] | None = (518, 518),
        device: str = "cuda",
    ):
        model = pipeline(task="depth-estimation", model=pretrained_config, device=device)
        super().__init__(model, image_shape)

    def predict(self, image: np.ndarray) -> Any:
        pil_img = Image.fromarray(image)
        return self.model(pil_img)["depth"]

    def process_and_save_labels(self, outputs: Any, filepath: str, save_path: str | None) -> None:
        orig_img = load_image(filepath, shape=None)
        orig_image_shape_xy = np.shape(orig_img)[:2][::-1]

        depth = cv2.resize(np.array(outputs), orig_image_shape_xy, interpolation=cv2.INTER_CUBIC)

        # TODO: properly compute+save depth values (instead of disparity)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)  # inferno coloring
        dst_file_depth = (
            filepath.replace(".png", "_depth.png")
            if save_path is None
            else os.path.join(save_path, os.path.basename(filepath).replace(".png", "_depth.png"))
        )

        cv2.imwrite(dst_file_depth, depth)
