from pathlib import Path
from typing import List, Tuple

try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    Document = None
    DOCX_AVAILABLE = False


def load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def ingest_plain_text(file_path: Path) -> str:
    return load_text_file(file_path)


def ingest_docx(file_path: Path) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed. Install it to ingest .docx files.")
    doc = Document(str(file_path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)

def ingest_directory(raw_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []

    for file_path in raw_dir.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix in [".txt", ".md"]:
            docs.append((file_path.name, ingest_plain_text(file_path)))
        elif suffix == ".docx":
            try:
                docs.append((file_path.name, ingest_docx(file_path)))
            except Exception:
                continue

    return docs