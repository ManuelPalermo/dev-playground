import itertools
import json
import os
import shutil
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ds_creator.clip_image_search import ClipModel
from ds_creator.utils import draw_text_as_image, glob_images, load_image


class SemanticDatasetCreator:
    def __init__(
        self,
        clip_model_name_weights: tuple[str, str] = ("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"),
        device: str = "cuda",
        image_shape: tuple[int, int] = (224, 224),
    ):
        self.image_shape = image_shape
        self.clip_model = ClipModel(*clip_model_name_weights, device=device)

    @torch.inference_mode()
    def compute_metadata_and_embeddings(
        self,
        file_path: str,
        compute_embedding: bool = True,
        compute_caption: bool = True,
    ) -> dict[str, Any]:
        image = load_image(file_path, shape=self.image_shape)

        metadata: dict[str, Any] = {
            "original_filepath": file_path,
            "model": f"{self.clip_model.model_version}+{self.clip_model.pretrained_weights}",
        }

        if compute_caption:
            assert "coca" in self.clip_model.model_version.lower(), "Captioning requires CoCa based model."
            caption = self.clip_model.compute_caption_from_image(image)
            metadata["caption"] = caption
            draw_text_as_image(text=caption, save_path=file_path.replace(".png", "_caption.png"))

        if compute_embedding:
            img_emb = self.clip_model.embed_image(image).tolist()
            metadata["embedding"] = img_emb

        return metadata

    def compute_metadata_and_embedding_directory(
        self,
        directory: str,
        compute_embedding: bool = True,
        compute_caption: bool = True,
        save_path: str | None = None,
    ):
        print("\n>> Running 'compute_metadata_and_embedding_directory'")

        for file_path in tqdm(glob_images(directory), desc="Computing metadata"):
            metadata = self.compute_metadata_and_embeddings(
                file_path,
                compute_embedding=compute_embedding,
                compute_caption=compute_caption,
            )

            if save_path is None:
                dst_file = file_path.replace(".png", "_metadata.json")
            else:
                dst_file = os.path.join(save_path, os.path.basename(file_path).replace(".png", "_metadata.json"))

            # save metadata
            with open(dst_file, "w", encoding="utf-8") as outfile_handle:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                json.dump(metadata, outfile_handle, indent=4)
            tqdm.write(f"Extracted file metadata to: {dst_file}")

    def text_search_directory(
        self,
        directory: str,
        text_query: str,
        top_k: int = 5,
        save_path: str | None = None,
    ) -> dict[str, float]:
        print("\n>> Running 'text_search_directory'")
        assert self.clip_model is not None, "Initialized clip model is required to run this operation."

        text_emb_query = self.clip_model.embed_text(text_query)

        scores: dict[str, float] = {}

        for file_path in tqdm(glob_images(directory), desc="Searching text query"):
            img_emb = self.clip_model.embed_image(load_image(file_path, shape=self.image_shape))
            img_score = self.clip_model.calculate_embed_similarity(text_emb_query, img_emb)
            scores[file_path] = img_score

        top_k_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

        print(f"Best matches for text query '{text_query}':")
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
        print("\n>> Running 'image_search_directory'")
        assert self.clip_model is not None, "Initialized clip model is required to run this operation."

        if isinstance(image_query, str) and os.path.isfile(image_query):
            img_emb_query = self.clip_model.embed_image(load_image(image_query, shape=self.image_shape))
        elif isinstance(image_query, np.ndarray):
            img_emb_query = self.clip_model.embed_image(image_query)
        else:
            raise ValueError("Invalid type for image_query", type(image_query))

        scores: dict[str, float] = {}

        for file_path in tqdm(glob_images(directory), desc="Searching img query"):
            img_emb = self.clip_model.embed_image(load_image(file_path, shape=self.image_shape))
            img_score = self.clip_model.calculate_embed_similarity(img_emb_query, img_emb)
            scores[file_path] = img_score

        top_k_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

        print(f"Best matches for image query {image_query if isinstance(image_query, str) else np.shape(image_query)}:")
        for idx, (file, score) in enumerate(top_k_scores.items()):
            print(f" -> {idx}:   Score: {score:.02f}  |  File: {file}")

            if save_path is not None:
                dst_file = os.path.join(save_path, os.path.basename(file))
                os.makedirs(save_path, exist_ok=True)
                shutil.copy2(file, dst_file)

        return top_k_scores

    def filter_similar_samples(
        self,
        directory: str,
        similarity_threshold: float = 0.90,
    ):
        print("\n>> Running 'filter_similar_samples'")
        file_embeddings: list[tuple[str, np.ndarray]] = []

        # get embeddings for all files (try to get from metadata if available, else compute)
        for file_path in glob_images(directory):
            metadata_file = file_path.replace(".png", "_metadata.json")
            if os.path.isfile(metadata_file):
                with open(metadata_file, encoding="utf-8") as json_handle:
                    embedding = np.array(json.load(json_handle)["embedding"], dtype=np.float32)
            else:
                image = load_image(file_path)
                embedding = self.clip_model.embed_image(image)

            file_embeddings.append((file_path, embedding))

        # go through all combination pairs and filter if similar
        removed_list: list[str] = []
        for idx1, idx2 in tqdm(list(itertools.combinations(range(len(file_embeddings)), 2)), desc="Filtering similar"):
            filename1, emb1 = file_embeddings[idx1]
            filename2, emb2 = file_embeddings[idx2]

            if filename2 in removed_list:
                continue

            similarity = cosine_similarity(emb1, emb2)

            if similarity > similarity_threshold:
                os.remove(filename2)
                metadata2_file = filename2.replace(".png", "_metadata.json")
                if os.path.isfile(metadata2_file):
                    os.remove(metadata2_file)
                removed_list.append(filename2)
                tqdm.write(
                    f"Deleted '{os.path.basename(filename2)}' as it had high similarity ({similarity.item():.02f}) "
                    f"to '{os.path.basename(filename1)}'."
                )
