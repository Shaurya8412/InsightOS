from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Backend URL
    INSIGHTOS_API_URL: str = "http://localhost:8000"

    # SQLite Database Configuration
    SQLITE_DB_URL: str = "sqlite:///insightos.db"

    # Gemini Configuration
    GEMINI_API_KEY: str = ""
    # text-embedding-004 is deprecated; gemini-embedding-001 is the current
    # production text embedding model. Change via EMBEDDING_MODEL env var.
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-3.6-flash"

    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "insightos_chunks"

    # RAG Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    DEFAULT_TOP_K: int = 5
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
