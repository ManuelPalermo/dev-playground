import os

import requests
from PIL import Image

from llm_app.llm_backends.llm_interface import LLMInterface

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # similar api interface to OpenAI, but free

SUPPORTED_MODELS = [
    "mistralai/mistral-7b-instruct",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-4-scout",
]


class OpenRouterClient(LLMInterface):
    """Client for OpenRouter API to interact with LLMs."""

    def __init__(
        self,
        api_url: str = OPENROUTER_API_URL,
        api_key: str | None = None,
        model_id: str = "mistralai/mistral-7b-instruct",
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

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = api_url

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "LLM api demo",
        }

    def __call__(
        self,
        text: str,
        image: str | Image.Image | None = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """Query the OpenRouter API with a prompt and return the model's response."""

        assert image is None, "Current LLM API does not support image queries."

        user_msg = {"role": "user", "content": text}

        messages = [*self.system_prompt, *self.history, user_msg]
        # print("\nMessage request:", message)

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        output = result["choices"][0]["message"]["content"]

        if self.history_num_turns > 0:
            # Append output to history
            self.history.append(user_msg)
            self.history.append({"role": "assistant", "content": output})

            # Truncate history (keep last n turns (prompt + response)
            self.history = self.history[-int(self.history_num_turns * 2) :]

            # try ot save conversation history to disc if not an anonymous chat
            self.save_conversation(img_history=True, sys_prompt_history=True)

        return output


def main() -> None:
    """Main function to demonstrate the usage of a LLM api and debug things."""

    # small example of how to use a LLM model from a free LLM API:
    api_model = OpenRouterClient(
        model_id="mistralai/mistral-7b-instruct",
        history_num_turns=5,
    )

    api_model.set_system_prompt("Respond like a pirate who enjoys sea shanties a bit too much...")

    output = api_model(
        text="Question: Can you give me a rough estimate of how many people are alive on Earth and "
        "how the population has evolved since the agricultural revolution?. \n\n",
        temperature=0.25,
        max_tokens=500,
    )
    print("\n--------------------\n")
    print("Model output:", output)
    print("---------//---------\n")

    output = api_model(
        text=(
            "Can you summarize the information into a text table, with columns:\nHuman milestone, Year, Population?\n\n"
        ),
        temperature=0.25,
        max_tokens=500,
    )
    print("\n--------------------\n")
    print("Model output:", output)
    print("---------//---------\n")

    # print model history
    history = api_model.show_history()
    print("\n--------------------")
    print("Model history:\n", history)
    print("---------//---------\n")

    # save model history to disc and then load it
    print("\n--------------------")
    conversation_name = "api_debug_history"
    api_model.set_conversation_name(conversation_name)
    api_model.save_conversation(img_history=False, sys_prompt_history=True)
    api_model.reset_history()
    api_model.reset_system_prompt()
    api_model.load_conversation(conversation_name, img_history=False, sys_prompt_history=True)
    # api_model.delete_conversation(conversation_name)
    print("---------//---------\n")


if __name__ == "__main__":
    main()
