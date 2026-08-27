from __future__ import annotations

from src.m2_search import SearchResult


class CrossEncoderReranker:
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, query: str, documents: list[dict | SearchResult], top_k: int = 3):
        results = [doc if isinstance(doc, SearchResult) else SearchResult(
            doc.get("text", ""), float(doc.get("score", 0)), doc.get("metadata", {}))
                   for doc in documents]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

