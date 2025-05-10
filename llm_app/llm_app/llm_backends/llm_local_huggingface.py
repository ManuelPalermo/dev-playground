from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.llava import LlavaForConditionalGeneration
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig

from llm_app.llm_backends.llm_interface import LLMInterface

SUPPORTED_MODELS = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-1.5-7b-hf",  # faster but not good with tools
    # "Qwen/Qwen2-VL-7B-Instruct",          # not working very well with images
]

LLM_MODEL_CLASS = {
    "llava-hf/llava-v1.6-mistral-7b-hf": LlavaNextForConditionalGeneration,
    "llava-hf/llava-1.5-7b-hf": LlavaForConditionalGeneration,
    "Qwen/Qwen2-VL-7B-Instruct": Qwen2VLForConditionalGeneration,
}


class OfflineHuggingFaceModel(LLMInterface):
    """Offline model wrapper with history and multimodal support."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str | torch.device | None = None,
        system_prompt: str | None = None,
        history_num_turns: int = 10,
        conversation_name: str | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            system_prompt=system_prompt,
            history_num_turns=history_num_turns,
            conversation_name=conversation_name,
        )
        assert model_id in SUPPORTED_MODELS

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            LLM_MODEL_CLASS[model_id]
            .from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            )
            .to(self.device)
        )
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    def __call__(
        self,
        text: str,
        image: str | Image.Image | None = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """Run the model with full history support."""
        # Append new turn to history
        if image is not None:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            }
        else:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }

        messages = [*self.system_prompt, *self.history, user_msg]

        # Prepare inputs
        if image is not None:
            inputs = self.prepare_image_text_inputs(messages, image)
        else:
            last_img = self.get_last_img_in_history()
            if last_img is not None:
                inputs = self.prepare_image_text_inputs(messages, last_img)
            else:
                inputs = self.prepare_text_only_inputs(messages)

        sample_params = (
            {
                "do_sample": True,
                "temperature": temperature,
            }
            if temperature > 0
            else {"do_sample": False}
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **sample_params,
        )

        output = self.processor.decode(output_ids[0][2:], skip_special_tokens=True)

        # trim model output template and keep only the actual response text
        if self.model_id == "llava-hf/llava-1.5-7b-hf":
            output = output.split("ASSISTANT:")[-1]
        elif self.model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
            output = output.split("[/INST] ")[-1]
        elif self.model_id == "Qwen/Qwen2-VL-7B-Instruct":
            output = output.split("assistant\n")[-1]

        # Append output to history
        if self.history_num_turns > 0:
            self.history.append(user_msg)
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": output.strip()}]})
            self.history_images.append(image)
            self.history_images.append(None)  # assistant has no image output

            # Truncate history (keep last n turns (prompt + response)
            self.history = self.history[-self.history_num_turns * 2 :]
            self.history_images = self.history_images[-self.history_num_turns * 2 :]

            # try ot save conversation history to disc if not an anonymous chat
            self.save_conversation(img_history=True, sys_prompt_history=True)

        return output.strip()

    def prepare_text_only_inputs(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Prepare inputs for text-only generation."""
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(
            text=prompt,
            return_tensors="pt",
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

    def prepare_image_text_inputs(
        self,
        messages: list[dict[str, Any]],
        image: str | Image.Image,
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for image-text generation."""
        # Get the last image for the current user message
        image = self.load_img(image)

        # Use most recent image for generation
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        return self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)


def main() -> None:
    """Main function to demonstrate the usage of a offline LLM model and debug things."""

    conversation_name = "local_debug_history"
    offline_model = OfflineHuggingFaceModel(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        # model_id="llava-hf/llava-1.5-7b-hf",
        device="cuda",
        history_num_turns=5,
        conversation_name=conversation_name,
        system_prompt="Respond like a pirate who enjoys sea shanties a bit too much...",
    )

    # image description
    img_url = "https://cdn.pixabay.com/photo/2018/08/04/11/30/draw-3583548_1280.png"

    output = offline_model(
        text="Give me a description of all the elements in the image, be as detailed as possible.\n\n",
        image=img_url,
        temperature=0.25,
        max_tokens=500,
    )
    print("\n--------------------")
    print("Model output (img/text):", output)
    print("---------//---------\n")

    # question answering
    output = offline_model(
        text=(
            "Unrelated to the previous question, but can you give me a rough estimate of how many people "
            "are alive on Earth and how the population has evolved since the agricultural revolution?. \n\n"
        ),
        image=None,
        temperature=0.25,
        max_tokens=500,
    )
    print("\n--------------------")
    print("Model output (text only):", output)
    print("---------//---------\n")

    # question answering
    output = offline_model(
        text=(
            "Can you summarize the information into a text table, with columns:\nHuman milestone, Year, Population?\n\n"
        ),
        image=None,
        temperature=0.25,
        max_tokens=500,
    )
    print("\n--------------------")
    print("Model output (text from history):", output)
    print("---------//---------\n")

    # print model history
    history = offline_model.show_history()
    print("\n--------------------")
    print("Model history:\n", history)
    print("---------//---------\n")

    # save model history to disc and then load it
    print("\n--------------------")
    offline_model.save_conversation(img_history=True, sys_prompt_history=True)
    offline_model.reset_history()
    offline_model.reset_system_prompt()
    offline_model.load_conversation(conversation_name, img_history=True, sys_prompt_history=True)
    print(offline_model.show_history())
    # offline_model.delete_conversation(conversation_name)
    print("---------//---------\n")


if __name__ == "__main__":
    main()
