from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    metadata: dict
    parent_id: str = "parent-0"


def load_documents(data_dir: str | None = None) -> list[dict]:
    root = Path(data_dir or "data")
    if not root.exists():
        return []
    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            except ImportError:
                continue
        else:
            continue
        documents.append({"text": text, "metadata": {"source": str(path)}})
    return documents


def chunk_hierarchical(text: str, metadata: dict | None = None, parent_size: int = 2048,
                       child_size: int = 256) -> tuple[list[Chunk], list[Chunk]]:
    metadata = dict(metadata or {})
    parents, children = [], []
    for parent_index, start in enumerate(range(0, len(text), parent_size)):
        parent_id = f"parent-{parent_index}"
        parent = Chunk(text[start:start + parent_size], metadata, parent_id)
        parents.append(parent)
        for child_start in range(0, len(parent.text), child_size):
            children.append(Chunk(parent.text[child_start:child_start + child_size],
                                   metadata, parent_id))
    return parents, children
