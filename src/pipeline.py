from __future__ import annotations

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker


class RAGPipeline:
    def __init__(self, *args, **kwargs):
        self.search = HybridSearch()
        self.reranker = CrossEncoderReranker()

    def build(self, documents: list[dict] | None = None):
        chunks = []
        for document in documents if documents is not None else load_documents():
            _, children = chunk_hierarchical(document["text"], document.get("metadata", {}))
            chunks.extend({"text": chunk.text, "metadata": chunk.metadata} for chunk in children)
        self.search.index(chunks)

    def query(self, question: str, top_k: int = 3):
        found = self.search.search(question)
        return self.reranker.rerank(question, found, top_k=top_k)

