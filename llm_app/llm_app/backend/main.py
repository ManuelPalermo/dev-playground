import io
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from llm_app.llm_backends import (
    LLM_CONVERSATION_STORE,
    SUPPORTED_LOCAL_MODELS,
    SUPPORTED_OPENROUTER_API_MODELS,
    LLMInterface,
    OfflineHuggingFaceModel,
    OpenRouterClient,
    RetrievalAugmentedGeneration,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_LENGHT = 25

BACKEND_MODELS = {
    "local_huggingface": SUPPORTED_LOCAL_MODELS,
    "openrouter_api": SUPPORTED_OPENROUTER_API_MODELS,
}


@app.get("/chat_list_conversations")
async def chat_list_conversations_endpoint() -> list[str]:
    """List all existing conversations stored on disk."""
    conversation_store = Path(LLM_CONVERSATION_STORE)

    if not conversation_store.exists():
        return []

    conversations = [f.name for f in conversation_store.iterdir() if f.is_dir()]
    return sorted(conversations)


@app.get("/chat_list_params")
async def chat_list_params_endpoint() -> dict[str, Any]:
    """List all existing conversations stored on disk."""
    params = {
        "history_lenght": HISTORY_LENGHT,
        "backend_models": BACKEND_MODELS,
    }
    return params


@lru_cache
def get_local_huggingface_model(model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf") -> OfflineHuggingFaceModel:
    """Load a local_huggingface model for offline inference."""
    assert model_id in BACKEND_MODELS["local_huggingface"], (
        f"Got model_id ({model_id}) which is not in available options: {BACKEND_MODELS['local_huggingface']}"
    )
    print(f"INFO: Loading Local Huggingface({model_id})")

    return OfflineHuggingFaceModel(model_id=model_id, history_num_turns=HISTORY_LENGHT)


@lru_cache
def get_openrouter_api_model(model_id: str = "mistralai/mistral-7b-instruct") -> OpenRouterClient:
    """Load an OpenRouter API model for inference."""
    assert model_id in BACKEND_MODELS["openrouter_api"], (
        f"Got model_id ({model_id}) which is not in available options: {BACKEND_MODELS['openrouter_api']}"
    )
    print(f"INFO: Loading OpenRouterClient({model_id})")
    return OpenRouterClient(model_id=model_id, history_num_turns=HISTORY_LENGHT)


@lru_cache
def get_rag_model(
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k_results: int = 5,
    similarity_score_threshold: float = 1.2,
) -> RetrievalAugmentedGeneration:
    """Load an OpenRouter API model for inference."""
    return RetrievalAugmentedGeneration(
        embeddings_model_name=embeddings_model_name,
        top_k_results=top_k_results,
        similarity_score_threshold=similarity_score_threshold,
    )


@app.post("/upload_file_for_rag")
async def upload_file_for_rag_endpoint(
    file: UploadFile = File(...),  # noqa: B008
    conversation_name: str = Form(...),
) -> dict[str, str | int]:
    """Handles the upload of a file for a Retrieval-Augmented Generation (RAG) endpoint."""

    rag = get_rag_model()

    conversation_name = conversation_name or "Anonymous"
    if file.filename is None:
        return {"status": "No file to upload"}

    # Save file to temp location
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # ingest + index file
    num_chunks = rag.ingest_and_index_file(
        filepath=file_path,
        namespace=conversation_name,
        save_original_file=True,
    )
    return {
        "status": "Uploaded",
        "filename": file.filename,
        "chunks": num_chunks,
    }


@app.post("/chat_openrouter_api")
async def chat_openrouter_api_endpoint(
    model_id: str = Form(...),
    conversation_name: str = Form(...),
    system_message: str = Form(...),
    message: str = Form(...),
    image_file: UploadFile | None = None,
    temperature: float = Form(0.1),
    max_tokens: int = Form(500),
) -> dict[str, str]:
    """Chat endpoint from an API LLM.

    Args:
        model_id: name of the LLM model to use.
        conversation_name: name of the conversation associated with the prompt.
        system_message: system prompt to style output generation.
        message: text prompt.
        image_file: image prompt (currently unsupported by the API).
        temperature: sampling temperature.
        max_tokens: maximum number of tokens to generate.
    """

    message = message.strip()
    model = get_openrouter_api_model(model_id)

    if image_file is not None:
        print("INFO: Image not supported for this API.")
        return {"response": "[Remote LLM API does not support image inputs!]"}

    return query_llm(
        model=model,
        conversation_name=conversation_name,
        system_message=system_message,
        message=message,
        image=None,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@app.post("/chat_local_huggingface")
async def chat_local_huggingface_endpoint(
    model_id: str = Form(...),
    conversation_name: str = Form(...),
    system_message: str = Form(...),
    message: str = Form(...),
    image_file: UploadFile | None = None,
    temperature: float = Form(0.1),
    max_tokens: int = Form(500),
) -> dict[str, str]:
    """Chat endpoint from a local_huggingface LLM.

    Args:
        model_id: name of the LLM model to use.
        conversation_name: name of the conversation associated with the prompt.
        system_message: system prompt to style output generation.
        message: text prompt
        image_file: image prompt.
        temperature: sampling temperature.
        max_tokens: maximum number of tokens to generate.

    """

    model = get_local_huggingface_model(model_id)

    # prepare img prompt if given
    image: Image.Image | None = None
    if image_file is not None:
        img_contents = await image_file.read()
        image = Image.open(io.BytesIO(img_contents)).convert("RGB")

    return query_llm(
        model=model,
        conversation_name=conversation_name,
        system_message=system_message,
        message=message,
        image=image,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def query_llm(  # noqa: PLR0912
    model: LLMInterface,
    conversation_name: str,
    system_message: str,
    message: str,
    image: Image.Image | None = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> dict[str, str]:
    """Query the LLM model with a prompt and return the model's response."""
    response = {}
    message = message.strip()
    context = ""

    # prepare system prompt if given
    if system_message:
        model.set_system_prompt(system_message)
    else:
        model.reset_system_prompt()

    # ---------- handle prompt ----------
    if message == "[CONVERSATION]":
        if conversation_name:
            # switch conversation name (load from disc if it already exists, otherwise create a new one)
            if model.conversation_name != conversation_name:
                model.load_conversation(conversation_name)
        else:
            # if no conversation name, then its a temporary chat which should not be saved
            model.set_conversation_name(None)

    elif message == "[RESET]":
        model.reset_history()
        model.set_conversation_name(None)
        # response = {"response": "[Chat history has been reset!]"}

    elif message == "[HISTORY]":
        if model.conversation_name:
            model.load_conversation(model.conversation_name)
            history = model.show_history()
        else:
            history = {}
        response = {
            "response": (f"[History (len={len(model.history) // 2} | max={model.history_num_turns})]:\n\n{history}")
        }

    else:
        # retrieve context using RAG and prepend it to the message
        rag = get_rag_model()
        context = rag.retrieve_context(query=message, namespace=conversation_name)
        prompt = f"CONTEXT:\n{context}\n\nUSER:\n{message}" if context else message

        output = model(text=prompt, image=image, temperature=temperature, max_tokens=max_tokens)
        response = {"response": output}
    # ------------------------------

    print("\n-------------------------")
    print("INFO: Received input query:")
    print(f"      model:                {model.__class__.__name__}({model.model_id})")
    print(f"      history len:          {len(model.history) // 2}")
    print(f"      conversation_name:    {conversation_name}")
    print(f"      system_message:       {system_message}")
    print(f"      temperature:          {temperature}")
    print(f"      max_tokens:           {max_tokens}")
    print(f"      context:              {context}")
    print(f"      message:              {message}")
    if image is not None:
        print(f"      image:                shape: {image.size}")
    else:
        print("       image:                None")
    print("      -------------")
    print("INFO: Produced output response:")
    print(f"Response:                   {response}")
    print("------------//-----------")

    return response
