import glob
import json
import os
import shutil
from typing import Any

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


class ClipModel:
    """Clip model handler.

    NOTE: uses model from https://github.com/mlfoundations/open_clip
    """

    def __init__(
        self,
        model_version="coca_ViT-L-14",
        pretrained_weights="mscoco_finetuned_laion2B-s13B-b90k",
        device: str = "cuda",
        precision: str = "fp32",
    ):
        self.device = device
        self.model_version = model_version
        self.pretrained_weights = pretrained_weights
        self.precision = precision

        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(
            model_version,
            pretrained=pretrained_weights,
            device=device,
            precision=precision,
        )
        self.tokenizer = open_clip.get_tokenizer(model_version)

    @torch.inference_mode()
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        proc_image = self.preprocessor(Image.fromarray(image)).to(self.device).unsqueeze(0)
        emb_img = self.model.encode_image(proc_image)
        return emb_img.cpu().numpy()

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        text_proc = self.tokenizer([text]).to(self.device)
        emb_text = self.model.encode_text(text_proc)
        return emb_text.cpu().numpy()

    @torch.inference_mode()
    def calculate_embed_similarity(self, emb_query: np.ndarray, emb_img: np.ndarray) -> float:
        emb_img /= np.linalg.norm(emb_img, axis=-1, keepdims=True)
        emb_query /= np.linalg.norm(emb_query, axis=-1, keepdims=True)

        probs = emb_query @ emb_img.T
        return float(np.squeeze(probs))

    @torch.inference_mode()
    def compute_image_text_similarity(self, image: np.ndarray, text: str) -> float:
        img_emb = self.embed_image(image)
        text_emb = self.embed_text(text)
        return self.calculate_embed_similarity(text_emb, img_emb)

    @torch.inference_mode()
    def compute_caption_from_image(self, image: np.ndarray) -> str:
        image_proc = self.preprocessor(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(self.device)

        generated = self.model.generate(image_proc)
        text = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        return text


class ClipImageSearch(ClipModel):
    def __init__(
        self,
        model_version="coca_ViT-L-14",
        pretrained_weights="mscoco_finetuned_laion2B-s13B-b90k",
        device: str = "cuda",
        precision: str = "fp32",
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(
            model_version=model_version,
            pretrained_weights=pretrained_weights,
            device=device,
            precision=precision,
        )
        self.image_size = image_size

    def load_image(self, file_path: str) -> np.ndarray:
        image = cv2.imread(file_path)
        image = cv2.resize(image, self.image_size, cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def image_extract_metadata(
        self, file_path: str, write_embedding: bool = True, write_caption: bool = True
    ) -> dict[str, Any]:
        image = self.load_image(file_path)

        caption = ""
        if write_caption:
            caption = self.compute_caption_from_image(image)

        img_emb = []
        if write_embedding:
            img_emb = self.embed_image(image).tolist()

        metadata = {
            "original_filepath": file_path,
            "model": f"{self.model_version}+{self.pretrained_weights}+{self.precision}",
            "caption": caption,
            "image_embedding": img_emb,
        }

        return metadata

    def compute_metadata_directory(
        self,
        directory: str,
        write_embedding: bool = True,
        write_caption: bool = True,
        save_path: str | None = None,
    ):
        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Computing metadata"):
            metadata = self.image_extract_metadata(
                file_path,
                write_embedding=write_embedding,
                write_caption=write_caption,
            )

            if save_path is None:
                dst_file = file_path.replace(".png", "_metadata.json")
            else:
                dst_file = os.path.join(save_path, os.path.basename(file_path).replace(".png", "_metadata.json"))
                with open(dst_file, "w", encoding="utf-8") as outfile_handle:
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    json.dump(metadata, outfile_handle, indent=4)
                tqdm.write(f"Saved file metadata to: {dst_file}")

    def clear_metadata_directory(self, directory: str):
        for file_path in tqdm(
            sorted(glob.glob(f"{directory}/**/*_emb.json", recursive=True)), desc="Cleaning metadata"
        ):
            os.remove(file_path)

    def text_search_directory(
        self,
        directory: str,
        text_query: str,
        top_k: int = 5,
        save_path: str | None = None,
    ) -> dict[str, float]:
        text_emb_query = self.embed_text(text_query)

        scores: dict[str, float] = {}

        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Searching text query"):
            img_emb = self.embed_image(self.load_image(file_path))
            img_score = self.calculate_embed_similarity(text_emb_query, img_emb)
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
        if isinstance(image_query, str) and os.path.isfile(image_query):
            img_emb_query = self.embed_image(self.load_image(image_query))
        elif isinstance(image_query, np.ndarray):
            img_emb_query = self.embed_image(image_query)
        else:
            raise ValueError("Invalid type for image_query", type(image_query))

        scores: dict[str, float] = {}

        for file_path in tqdm(sorted(glob.glob(f"{directory}/**/*.png", recursive=True)), desc="Searching img query"):
            img_emb = self.embed_image(self.load_image(file_path))
            img_score = self.calculate_embed_similarity(img_emb_query, img_emb)
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
