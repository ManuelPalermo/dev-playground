import numpy as np
import open_clip
import torch
from PIL import Image


class ClipModel:
    """Clip model handler.

    NOTE: uses model from https://github.com/mlfoundations/open_clip
    """

    def __init__(
        self,
        model_version="coca_ViT-L-14",
        pretrained_weights="mscoco_finetuned_laion2B-s13B-b90k",
        device: str = "cuda",
    ):
        self.device = device
        self.model_version = model_version
        self.pretrained_weights = pretrained_weights

        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(
            model_version,
            pretrained=pretrained_weights,
            device=device,
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
