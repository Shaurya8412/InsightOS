import os
import subprocess
import time
import urllib.request
import json
import pytest
from playwright.sync_api import sync_playwright
from src.core.config import settings

def wait_for_url(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

@pytest.mark.integration
def test_streamlit_e2e_flow():
    # 0. Clean DB and Qdrant to ensure test isolation
    from src.core.database import SessionLocal
    from src.models.db_models import Document
    from qdrant_client import QdrantClient
    
    db = SessionLocal()
    try:
        db.query(Document).delete()
        db.commit()
    finally:
        db.close()
        
    try:
        qc = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        if qc.collection_exists(settings.QDRANT_COLLECTION_NAME):
            qc.delete_collection(settings.QDRANT_COLLECTION_NAME)
    except Exception as e:
        print("Cleanup collections failed:", e)

    # 1. Start FastAPI backend
    backend_proc = subprocess.Popen(
        [".venv/Scripts/python.exe", "-m", "uvicorn", "src.main:app", "--port", "8000", "--host", "127.0.0.1"]
    )
    
    # 2. Start Streamlit frontend
    frontend_proc = subprocess.Popen(
        [".venv/Scripts/python.exe", "-m", "streamlit", "run", "src/frontend/app.py", "--server.port", "8501", "--server.address", "127.0.0.1"]
    )
    
    try:
        # Wait for FastAPI
        assert wait_for_url("http://127.0.0.1:8000/health"), "FastAPI backend failed to start"
        print("Backend started and healthy.")
        
        # Wait for Streamlit
        time.sleep(5)  # Wait extra for Streamlit server init
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Go to streamlit page
            page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
            page.goto("http://127.0.0.1:8501")
            
            # Allow page loading
            page.wait_for_timeout(3000)
            
            # Check title
            assert "InsightOS" in page.title()
            print("Streamlit UI opened successfully.")
            
            # Upload document
            try:
                file_input = page.locator("input[type='file']")
                file_input.set_input_files("test_document.pdf")
                print("Uploading test_document.pdf...")
                
                # Wait for upload status to change to INDEXED
                page.wait_for_selector("text=INDEXED", timeout=30000)
            except Exception as e:
                page.screenshot(path="tests/integration/failure_screenshot.png")
                print("Screenshot saved to tests/integration/failure_screenshot.png")
                raise e
            print("Ingestion center: INDEXED status reached.")
            
            # Verify database entry using FastAPI endpoint
            req = urllib.request.Request("http://127.0.0.1:8000/api/v1/documents")
            with urllib.request.urlopen(req) as resp:
                docs = json.loads(resp.read().decode())
                uploaded_doc = next((d for d in docs if d["filename"] == "test_document.pdf"), None)
                assert uploaded_doc is not None
                assert uploaded_doc["status"] == "indexed"
                assert uploaded_doc["chunk_count"] > 0
                print(f"Verified SQLite document state: {uploaded_doc['status']}")
            
            # Query input
            query_input = page.get_by_placeholder("e.g. What is the architecture of RAG?")
            query_input.fill("What embedding model does InsightOS use?")
            page.screenshot(path="tests/integration/before_query_screenshot.png")
            
            # Submit query by pressing Enter
            query_input.press("Enter")
            print("Pressed Enter on query input...")
            
            # Wait for assistant response bubble
            page.wait_for_selector("div.chat-assistant", timeout=30000)
            page.screenshot(path="tests/integration/after_query_screenshot.png")
            
            # Verify answer text contains correct model info
            content = page.content()
            assert "text-embedding-004" in content.lower() or "gemini-embedding-001" in content.lower(), "Answer not grounded or incorrect model mentioned"
            print("Query answer verified.")
            
            # Verify citation expands
            citation_expander = page.locator("summary", has_text="test_document.pdf").first
            assert citation_expander.is_visible()
            citation_expander.click()
            page.wait_for_timeout(1000)
            assert "InsightOS" in page.content(), "Citation text not found in UI expander"
            print("Citation UI checked and verified.")
            
            # Delete document
            delete_btn = page.get_by_role("button", name="Delete 🗑️").first
            assert delete_btn.is_visible()
            delete_btn.click()
            print("Click delete button...")
            
            # Wait for document to disappear from library UI
            page.wait_for_timeout(5000)
            assert page.get_by_role("button", name="Delete 🗑️").count() == 0, "Document Delete button still exists in library UI"
            print("Library UI deletion verified.")
            
            # Verify in SQLite via FastAPI
            with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/documents") as resp:
                docs = json.loads(resp.read().decode())
                uploaded_doc = next((d for d in docs if d["filename"] == "test_document.pdf"), None)
                assert uploaded_doc is None, "SQLite record still exists after deletion"
            print("SQLite deletion verified.")
            
            # Query again to verify post-deletion behavior
            query_input = page.get_by_placeholder("e.g. What is the architecture of RAG?")
            query_input.fill("What embedding model does InsightOS use?")
            query_input.press("Enter")
            print("Submitted post-deletion query...")
            
            # Wait for second assistant response bubble (index 1)
            page.wait_for_selector("div.chat-assistant >> nth=1", timeout=30000)
            
            assert "cannot answer" in page.content().lower() or "not found" in page.content().lower() or "no context" in page.content().lower(), "Post-deletion query response not fallback"
            print("Post-deletion query verified.")
            
            browser.close()
            
    finally:
        # Cleanup
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("Subprocesses stopped.")
