from pathlib import Path
from typing import List, Tuple


def load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def ingest_directory(raw_dir: Path) -> List[Tuple[str, str]]:
    docs = []
    for file_path in raw_dir.iterdir():
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in [".txt", ".md"]:
            docs.append((file_path.name, load_text_file(file_path)))
    return docs
