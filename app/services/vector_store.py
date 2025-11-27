# app/services/vector_store.py
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

load_dotenv()


class VectorStore:
    """
    Handles:
    - Creating / reusing Chroma collections
    - Ingesting structured service data into embeddings
    - Retrieving relevant items
    """

    def __init__(self, persist_directory: str | None = None):
        self.persist_dir = persist_directory or settings.chroma_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Google embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", google_api_key=settings.google_api_key
        )

        # Load existing (or create if not exists)
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="company_services",
        )

    # -----------------------------------
    # Convert service dict → Document
    # -----------------------------------
    def _service_to_document(self, s: Dict) -> Document:
        content = (
            f"{s['item']}. {s['description']}. "
            f"Best for: {s['best_for']}. "
            f"Features: {s['features']}. "
            f"Price: {s['price']}"
        )

        metadata = {
            "item": s["item"],
            "service_type": s["service_type"],
            "price": s["price"],
        }

        return Document(page_content=content, metadata=metadata)

    # -----------------------------------
    # Ingest new services
    # -----------------------------------
    def ingest_services(self, services: List[Dict]):
        docs = [self._service_to_document(s) for s in services]

        # Rebuild collection completely
        self.store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="company_services",
        )

        try:
            self.store.persist()
        except Exception:
            pass  # some versions persist automatically

        return True

    # -----------------------------------
    # Retrieval interface
    # -----------------------------------
    def retriever(self, k: int = 4):
        return self.store.as_retriever(search_kwargs={"k": k})

    def get_top_documents(self, query: str, k: int = 4) -> List[Document]:
        # return self.retriever(k).get_relevant_documents(query)
        return self.retriever(k).invoke(query)
