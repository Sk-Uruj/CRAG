from typing import List

from backend.models import DocumentChunk


def simple_chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap

    return chunks


def chunk_documents(docs: List[tuple[str, str]], chunk_size: int = 800, overlap: int = 120) -> List[DocumentChunk]:
    all_chunks: List[DocumentChunk] = []
    for source_name, text in docs:
        pieces = simple_chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_name}_chunk_{idx+1}",
                    source=source_name,
                    text=piece,
                    metadata={
                        "source": source_name,
                        "chunk_index": idx + 1,
                        "total_chunks": len(pieces),
                    },
                )
            )
    return all_chunks
