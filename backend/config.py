from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "ThoughtsTracer AI"
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"

    CHROMA_COLLECTION_NAME: str = "thoughts_tracer_docs"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


settings = Settings()


def ensure_directories() -> None:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
