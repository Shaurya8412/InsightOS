import os
from src.core.config import Settings

def test_settings_load(monkeypatch):
    # Provide required/dummy env vars to override defaults or supply missing ones if needed
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    
    settings = Settings()
    
    assert settings.GEMINI_API_KEY == "test-key"
    assert settings.QDRANT_API_KEY == "qdrant-test"
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.EMBEDDING_MODEL == "gemini-embedding-001"
    assert settings.LLM_MODEL == "gemini-3.6-flash"
