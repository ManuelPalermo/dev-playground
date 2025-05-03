import abc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

LLM_CONVERSATION_STORE = f"{Path.home()}/dev-playground/llm_app/llm_app/.llm_conversation_history/"


def download_image(img_url: Path | str, save_dir: Path | str) -> str:
    """Download image and save it to disk."""
    response = requests.get(str(img_url), timeout=10)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    file_hash = hashlib.md5(image.tobytes()).hexdigest()  # noqa: S324
    outfile = f"{save_dir}/{file_hash}.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(filename=outfile, img=image)
    return outfile


class LLMInterface:
    """LLM model interface."""

    LOCAL_IMG_STORE = tempfile.gettempdir()
    LLM_CONVERSATION_STORE = LLM_CONVERSATION_STORE

    def __init__(
        self,
        model_id: str,
        system_prompt: str | None = None,
        history_num_turns: int = 10,
        conversation_name: str | None = None,
    ) -> None:
        self.model_id = model_id

        self.conversation_name: str | None = None
        self.set_conversation_name(conversation_name)

        self.system_prompt: list[dict[str, Any]] = []
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)

        self.history_num_turns = history_num_turns
        self.history: list[dict[str, Any]] = []
        self.history_images: list[Image.Image | None] = []

    @abc.abstractmethod
    def __call__(
        self,
        text: str,
        image: str | Image.Image | None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Run an LLM model with a given text and image (optional) query and get a response."""
        raise NotImplementedError

    def load_img(self, img: str | Path | Image.Image) -> Image.Image:
        """Load image from path."""
        if isinstance(img, Image.Image):
            return img

        if "http" in str(img):
            img_path = download_image(str(img), self.LOCAL_IMG_STORE)
            return Image.open(str(img_path)).convert("RGB")

        return Image.open(str(img)).convert("RGB")

    def get_last_img_in_history(self) -> Image.Image | None:
        """Get latest valid img from history."""
        for img in reversed(self.history_images):
            if img is not None:
                return img
        return None

    def show_history(self) -> str:
        """Returns the conversation history in a readable string."""
        print("INFO: requested model history.")
        history = [*self.system_prompt, *self.history]
        history_str = "\n\n".join([f"{hist}" for hist in history])
        return history_str

    def reset_history(self) -> None:
        """Reset the conversation history."""
        print("INFO: clearing model history.")
        self.system_prompt = []
        self.history = []
        self.history_images = []

    def set_system_prompt(self, message: str) -> None:
        """Sets a system prompt to style the LLM generation."""
        print(f"INFO: setting system prompt. {message}")
        self.system_prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": str(message).strip()}],
            }
        ]

    def reset_system_prompt(self) -> None:
        """Reset the system prompt history."""
        print("INFO: clearing system prompt history.")
        self.system_prompt = []

    def set_conversation_name(self, conversation_name: str | None) -> None:
        """Sets a conversation name."""
        if self.conversation_name != conversation_name or conversation_name is None:
            self.reset_history()

        self.conversation_name = conversation_name

        # create empty conversation folder
        if conversation_name is not None:
            (Path(self.LLM_CONVERSATION_STORE) / conversation_name).mkdir(exist_ok=True, parents=True)

    def save_conversation(self, *, img_history: bool = True, sys_prompt_history: bool = True) -> None:
        """Saves current history to file."""

        if self.conversation_name is None:
            print("[INFO] No conversation name has been set, anonymous conversation will not saved.")
            return

        (Path(self.LLM_CONVERSATION_STORE) / self.conversation_name).mkdir(exist_ok=True, parents=True)

        # save text history
        history_file_chat = Path(self.LLM_CONVERSATION_STORE) / self.conversation_name / "chat_history.json"
        with history_file_chat.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        # save system prompt
        if sys_prompt_history:
            system_file_chat = Path(self.LLM_CONVERSATION_STORE) / self.conversation_name / "system_prompt.json"
            with system_file_chat.open("w", encoding="utf-8") as f:
                json.dump(self.system_prompt, f, indent=4)

        # save images
        if img_history:
            img_dir = Path(self.LLM_CONVERSATION_STORE) / self.conversation_name / "images"
            img_dir.mkdir(exist_ok=True, parents=True)
            for idx, img in enumerate(self.history_images):
                if img is not None:
                    self.load_img(img).save(img_dir / f"{idx}.png")

        print(f"INFO: Saved conversation '{self.conversation_name}' to file: {history_file_chat}")

    def load_conversation(
        self, conversation_name: str, *, img_history: bool = True, sys_prompt_history: bool = True
    ) -> None:
        """Load history from file."""
        self.set_conversation_name(conversation_name=conversation_name)

        conversation_path = Path(self.LLM_CONVERSATION_STORE) / conversation_name
        history_file_chat = conversation_path / "chat_history.json"
        if not history_file_chat.exists():
            print("[INFO] conversation does not yet exist, so creating a new one.")
            return

        # load text history
        with history_file_chat.open("r", encoding="utf-8") as f:
            self.history = json.load(f)

        # load system prompt
        if sys_prompt_history:
            system_file_chat = conversation_path / "system_prompt.json"
            with system_file_chat.open("r", encoding="utf-8") as f:
                self.system_prompt = json.load(f)

        # load images
        if img_history:
            self.history_images = [None for _ in range(len(self.history))]
            img_dir = conversation_path / "images"
            for img_filename in Path(str(img_dir)).iterdir():
                img = self.load_img(img_filename)
                img_idx = int(img_filename.stem.replace(".png", ""))
                self.history_images[img_idx] = img

        print(f"INFO: Loaded conversation '{self.conversation_name}' from file: {history_file_chat}")
        print(self.show_history())
        print("-----//-----")

    def delete_conversation(self, conversation_name: str) -> None:
        """Deletes the conversation from disc."""
        print(f"INFO: Deleting conversation named {conversation_name} from dir {self.LLM_CONVERSATION_STORE}.")
        shutil.rmtree(Path(self.LLM_CONVERSATION_STORE) / conversation_name)
