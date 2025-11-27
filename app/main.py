from fastapi import FastAPI
from app.api import router

app = FastAPI(title="LLM Recommender", version="0.1")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}
