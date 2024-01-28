import glob
import json
import os
import shutil
from typing import Any

import cv2
import numpy as np
import torch
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from ds_creator.clip_image_search import ClipModel
from tqdm import tqdm


class SemanticDatasetCreator:
    def __init__(
        self,
        clip_model_name_weights: tuple[str, str] | None = ("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"),
        grounding_sam_model=None,
        device: str = "cuda",
        image_shape: tuple[int, int] = (224, 224),
    ):
        self.clip_model = ClipModel(*clip_model_name_weights, device=device) if clip_model_name_weights else None

        self.grounding_sam_model = (
            GroundedSAM(ontology=CaptionOntology({"dog": "dog", "cat": "cat", "birds": "birds"}))
            if grounding_sam_model
            else None
        )

        self.image_shape = image_shape

    @torch.inference_mode()
    def compute_metadata_and_autolabel(
        self,
        file_path: str,
        compute_embedding: bool = True,
        compute_caption: bool = True,
        compute_instance_labels: bool = False,
    ) -> tuple[dict[str, Any], None | list[np.ndarray]]:
        image = self.load_image(file_path, shape=self.image_shape)

        metadata: dict[str, Any] = {
            "original_filepath": file_path,
        }

        if compute_caption:
            assert self.clip_model is not None, "Initialized clip model is required to 'compute_caption'."
            caption = self.clip_model.compute_caption_from_image(image)
            metadata["caption"] = {
                "model": f"{self.clip_model.model_version}+{self.clip_model.pretrained_weights}",
                "caption": caption,
            }

        semseg_mask = None
        if compute_instance_labels:
            assert (
                self.grounding_sam_model is not None
            ), "Initialized Grounding-SAM model is required to 'compute_instance_segmentation'."

            detections = self.grounding_sam_model.predict(input=image)

            # parse bbox 2d
            box_metadata = {}
            for idx, box_2d in enumerate(detections.xyxy):
                confidence = detections.confidence[idx] if detections.confidence is not None else -1
                if detections.class_id is not None:
                    class_id = detections.class_id[idx]
                    class_name = self.grounding_sam_model.ontology.classes()[class_id]
                else:
                    class_id = -1
                    class_name = ""

                box_metadata[f"box_{idx}"] = {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "confidence": float(confidence),
                    "box": box_2d.tolist(),
                }
            metadata["instance"] = {
                "model": f"{self.grounding_sam_model}",
                "onthology": self.grounding_sam_model.ontology.promptMap,
                "bbox_2d": box_metadata,
            }

            # parse semseg mask
            if detections.mask is not None and detections.mask.size > 0:
                orig_image_shape_xy = np.shape(self.load_image(file_path, shape=None))[:2][::-1]

                # convert bool masks to [0,255] uint8 and resize to original img shape
                semseg_mask = np.transpose(detections.mask.astype(np.uint8) * 255, axes=(1, 2, 0))
                semseg_mask = cv2.resize(semseg_mask, orig_image_shape_xy, interpolation=cv2.INTER_NEAREST)
                if detections.mask.shape[0] == 1:
                    semseg_mask = semseg_mask[..., None]
                semseg_mask = [semseg_mask[..., idx] for idx in range(semseg_mask.shape[-1])]

        if compute_embedding:
            assert self.clip_model is not None, "Initialized clip model is required to 'compute_embedding'."
            img_emb = self.clip_model.embed_image(image).tolist()
            metadata["embedding"] = {
                "model": f"{self.clip_model.model_version}+{self.clip_model.pretrained_weights}",
                "embedding": img_emb,
            }

        return metadata, semseg_mask

    def compute_autolabel_directory(
        self,
        directory: str,
        compute_embedding: bool = True,
        compute_caption: bool = True,
        compute_instance_labels: bool = True,
        save_path: str | None = None,
    ):
        print(">> Running 'compute_autolabel_directory'")

        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Computing metadata"):
            metadata, semseg_mask = self.compute_metadata_and_autolabel(
                file_path,
                compute_embedding=compute_embedding,
                compute_caption=compute_caption,
                compute_instance_labels=compute_instance_labels,
            )

            if save_path is None:
                dst_file = file_path.replace(".png", "_metadata.json")
            else:
                dst_file = os.path.join(save_path, os.path.basename(file_path).replace(".png", "_metadata.json"))

            # save semseg mask
            if semseg_mask is not None:
                for idx, mask in enumerate(semseg_mask):
                    semseg_dst_file = dst_file.replace("_metadata.json", f"_semseg_mask_{idx}.png")
                    cv2.imwrite(semseg_dst_file, mask)
                    metadata["instance"]["bbox_2d"][f"box_{idx}"]["mask"] = semseg_dst_file

                semseg_dst_file = dst_file.replace("_metadata.json", "_semseg_mask_stack.png")
                stacked_mask = np.stack(semseg_mask).max(axis=0)
                cv2.imwrite(semseg_dst_file, stacked_mask)
                metadata["instance"]["full_semseg_mask"] = semseg_dst_file

            # save metadata
            with open(dst_file, "w", encoding="utf-8") as outfile_handle:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                json.dump(metadata, outfile_handle, indent=4)
            tqdm.write(f"Extracted file metadata to: {dst_file}")

    def clear_directory(self, directory: str, clear_search: bool = False, clear_metadata_and_labels: bool = True):
        print(">> Running 'clear_directory'")

        search_files = []

        if clear_search:
            search_files.extend(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)))

        if clear_metadata_and_labels or clear_search:
            search_files.extend(sorted(glob.glob(f"{directory}/**/*_metadata.json", recursive=True)))
            search_files.extend(sorted(glob.glob(f"{directory}/**/*_semseg_mask.png", recursive=True)))

        for file_path in tqdm(search_files, desc="Cleaning metadata"):
            os.remove(file_path)

    def text_search_directory(
        self,
        directory: str,
        text_query: str,
        top_k: int = 5,
        save_path: str | None = None,
    ) -> dict[str, float]:
        print(">> Running 'text_search_directory'")
        assert self.clip_model is not None, "Initialized clip model is required to run this operation."

        text_emb_query = self.clip_model.embed_text(text_query)

        scores: dict[str, float] = {}

        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Searching text query"):
            img_emb = self.clip_model.embed_image(self.load_image(file_path, shape=self.image_shape))
            img_score = self.clip_model.calculate_embed_similarity(text_emb_query, img_emb)
            scores[file_path] = img_score

        top_k_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

        print(f"\nBest matches for text query '{text_query}':")
        for idx, (file, score) in enumerate(top_k_scores.items()):
            print(f" -> {idx}:   Score: {score:.02f}  |  File: {file}")

            if save_path is not None:
                dst_file = os.path.join(save_path, os.path.basename(file))
                os.makedirs(save_path, exist_ok=True)
                shutil.copy2(file, dst_file)

        return top_k_scores

    def image_search_directory(
        self,
        directory: str,
        image_query: np.ndarray | str,
        top_k: int = 5,
        save_path: str | None = None,
    ) -> dict[str, float]:
        print(">> Running 'image_search_directory'")
        assert self.clip_model is not None, "Initialized clip model is required to run this operation."

        if isinstance(image_query, str) and os.path.isfile(image_query):
            img_emb_query = self.clip_model.embed_image(self.load_image(image_query, shape=self.image_shape))
        elif isinstance(image_query, np.ndarray):
            img_emb_query = self.clip_model.embed_image(image_query)
        else:
            raise ValueError("Invalid type for image_query", type(image_query))

        scores: dict[str, float] = {}

        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Searching img query"):
            img_emb = self.clip_model.embed_image(self.load_image(file_path, shape=self.image_shape))
            img_score = self.clip_model.calculate_embed_similarity(img_emb_query, img_emb)
            scores[file_path] = img_score

        top_k_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

        print(
            f"\nBest matches for image query {image_query if isinstance(image_query, str) else np.shape(image_query)}:"
        )
        for idx, (file, score) in enumerate(top_k_scores.items()):
            print(f" -> {idx}:   Score: {score:.02f}  |  File: {file}")

            if save_path is not None:
                dst_file = os.path.join(save_path, os.path.basename(file))
                os.makedirs(save_path, exist_ok=True)
                shutil.copy2(file, dst_file)

        return top_k_scores

    @staticmethod
    def load_image(file_path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if shape is not None:
            image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)

        return image
