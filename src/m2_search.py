from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    def __init__(self, *args, **kwargs):
        self.documents: list[dict] = []

    def index(self, chunks: list[dict]) -> None:
        self.documents = list(chunks)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for item in self.documents:
            words = set(re.findall(r"\w+", item.get("text", "").lower()))
            score = len(query_words & words) / max(len(query_words), 1)
            scored.append(SearchResult(item.get("text", ""), score, item.get("metadata", {})))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

