from fastapi import FastAPI
from src.core.config import settings
from src.api.routes import router as api_router

app = FastAPI(title="InsightOS MVP")

app.include_router(api_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
