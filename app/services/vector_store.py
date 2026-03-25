import os
from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
except ImportError:
    chromadb = None
    HuggingFaceEmbeddings = None
    Chroma = None

class VectorStoreService:
    def __init__(self, persist_directory: Optional[str] = "./chromadb_store"):
        self.persist_directory = persist_directory
        self.collection_name = "documents"

        if chromadb is None or HuggingFaceEmbeddings is None or Chroma is None:
            self.store = None
            return

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.store = Chroma(collection_name=self.collection_name, embedding_function=self.embeddings, persist_directory=self.persist_directory)

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        if self.store is None:
            return 0

        self.store.add_texts(texts=texts, metadatas=metadatas or [{} for _ in texts], ids=ids)
        self.store.persist()
        return len(texts)

    def similarity_search(self, query: str, k: int = 3):
        if self.store is None:
            return []

        results = self.store.similarity_search_with_score(query, k=k)
        output = []
        for doc, score in results:
            output.append({"id": doc.metadata.get("id"), "text": doc.page_content, "score": float(score)})
        return output

    def clear_collection(self):
        if self.store is None:
            return
        self.store.delete_collection()
        self.store = Chroma(collection_name=self.collection_name, embedding_function=self.embeddings, persist_directory=self.persist_directory)
