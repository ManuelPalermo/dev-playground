import hashlib
import shutil
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_STORE = Path(f"{Path.home()}/dev-playground/llm_app/llm_app/.conversation_history/")


EXTENSION_TO_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    # NOTE: following formats could probably be optimized with specific loaders
    ".json": TextLoader,
    ".md": TextLoader,
    ".html": TextLoader,
}


def load_file(filepath: Path | str) -> list[Document]:
    """Load and parse file using corresponding loader."""
    filepath = Path(filepath)
    loader = EXTENSION_TO_LOADER_MAP[filepath.suffix]
    return loader(str(filepath)).load()


class RetrievalAugmentedGeneration:
    """Retrieval Augmented Generation."""

    def __init__(
        self,
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_results: int = 5,
        similarity_score_threshold: float = 1.2,
    ) -> None:
        self._embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self._top_k_results = top_k_results
        self._similarity_score_threshold = similarity_score_threshold

    def ingest_and_index_file(
        self,
        filepath: Path | str,
        namespace: str = "default",
        *,
        save_original_file: bool = True,
    ) -> int:
        """Ingests a file, chunks it, generates embeddings from the chunks, and indexes them into a FAISS vector db."""

        namespace_store = VECTOR_DB_STORE / namespace
        original_files_store = namespace_store / "files"
        namespace_store.mkdir(exist_ok=True, parents=True)
        original_files_store.mkdir(exist_ok=True, parents=True)

        filename = Path(filepath).name
        if save_original_file:
            shutil.copy2(str(filepath), original_files_store)
            file_hash = hashlib.md5(Path(filepath).read_bytes()).hexdigest()  # noqa: S324
            dst_orig_filepath = original_files_store / f"{file_hash}{Path(filepath).suffix}"
            shutil.move(str(original_files_store / filename), dst_orig_filepath)

        data = load_file(filepath)
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents(data)

        if (namespace_store / "index.faiss").exists():
            db = FAISS.load_local(
                folder_path=str(namespace_store),
                embeddings=self._embeddings_model,
                allow_dangerous_deserialization=True,
            )
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, self._embeddings_model)

        db.save_local(str(namespace_store))
        return len(chunks)

    def retrieve_context(self, query: str, namespace: str = "default") -> str:
        """Loads vectorstore and retrieves similar chunks."""
        namespace_store = VECTOR_DB_STORE / namespace

        if not (namespace_store / "index.faiss").exists():
            print(f"[RAG] No vector store found for namespace: {namespace}")
            return ""

        db = FAISS.load_local(
            folder_path=str(namespace_store),
            embeddings=self._embeddings_model,
            allow_dangerous_deserialization=True,
        )
        results = db.similarity_search_with_score(query, k=self._top_k_results)
        return "\n\n".join([doc.page_content for doc, score in results if score < self._similarity_score_threshold])
