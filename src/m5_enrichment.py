from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict


def enrich_chunks(chunks: list[dict], *args, **kwargs) -> list[EnrichedChunk]:
    return [EnrichedChunk(item.get("text", ""), dict(item.get("metadata", {}))) for item in chunks]

