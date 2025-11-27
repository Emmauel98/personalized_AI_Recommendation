# LLM Recommender — Full Project (Python, FastAPI, LangChain, Chroma)
# Author: Senior-style, production-minded, simple to run
# Files included below — save each section into its filename as shown.

####################################################################
# File: README.md
####################################################################
# LLM Recommender

A minimal, professional, production-minded Python project that:
- Accepts a user request via a FastAPI endpoint
- Retrieves relevant content from a local Chroma vector store
- Uses a cloud LLM (pluggable: OpenAI / Google Gemini / local) via LangChain
- Returns a recommendation strictly based on the provided content feed

This repository is deliberately small, modular, and easy to extend for any
service domain (transport, products, real-estate, healthcare, etc.).

---

### Quick start (local demo)

1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Add a `.env` file with provider keys. Example for OpenAI:

```
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai   # or 'google' for Gemini (if configured)
CHROMA_DIR=./chroma_db
```

3. Ingest demo documents (service catalog) into the vector store:

```bash
python scripts/ingest_demo.py
```

4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

5. Example request (curl):

```bash
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" \
  -d '{"question":"Best option to travel Lagos to Abuja?","k":3}'
```

---

### Project structure

```
llm-recommender/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── api.py                  # API route(s)
│   ├── config.py               # env/config loader
│   ├── services/               # small service layer
│   │   ├── llm_service.py      # LLM wrapper / chain builder
│   │   ├── vector_store.py     # Chroma wrapper
│   │   └── prompts.py          # prompt templates
│   └── models.py               # pydantic request/response models
├── scripts/
│   └── ingest_demo.py          # loads demo content into Chroma
├── requirements.txt
├── README.md
└── .env.example
```

---

####################################################################
# File: requirements.txt
####################################################################

fastapi
uvicorn[standard]
python-dotenv
langchain>=0.0.200
chroma-hnswlib
sentence-transformers
pydantic
openai
# If you prefer Google Gemini via google-generativeai:
# google-generativeai

####################################################################
# File: .env.example
####################################################################

# Choose llm provider: openai | google
LLM_PROVIDER=openai
OPENAI_API_KEY=
CHROMA_DIR=./chroma_db

####################################################################
# File: app/config.py
####################################################################
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    chroma_dir: str = "./chroma_db"
    # LLM config
    llm_model: str = "gpt-4o-mini"  # change as desired
    temperature: float = 0.0

    class Config:
        env_file = ".env"

settings = Settings()

####################################################################
# File: app/models.py
####################################################################
from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    question: str
    k: Optional[int] = 4  # number of retrieved docs

class Recommendation(BaseModel):
    recommendation: str
    explanation: str
    retrieved: List[str]

class RecommendResponse(BaseModel):
    success: bool
    result: Recommendation | None = None
    error: str | None = None

####################################################################
# File: app/services/vector_store.py
####################################################################
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
from app.config import settings

# Simple wrapper around Chroma local
class VectorStore:
    def __init__(self, persist_directory: str | None = None):
        persist_directory = persist_directory or settings.chroma_dir
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        # Using a small HuggingFace embedding model (sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.store = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

    def from_texts(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None):
        """Overwrite or create collection from texts (useful for demo ingest)."""
        collection = Chroma.from_texts(texts=texts, embedding=self.embeddings, persist_directory=settings.chroma_dir)
        collection.persist()
        self.store = collection
        return collection

    def retriever(self, k: int = 4):
        return self.store.as_retriever(search_kwargs={"k": k})

    def get_top_documents(self, query: str, k: int = 4):
        retriever = self.retriever(k=k)
        docs = retriever.get_relevant_documents(query)
        return [d.page_content for d in docs]


####################################################################
# File: app/services/prompts.py
####################################################################
from langchain.prompts import PromptTemplate

BASE_PROMPT = """
You are a strict recommendation engine.

RULES:
- You MUST ONLY use the items in the `context` section below.
- Do NOT hallucinate, invent services, or add assumptions not present.
- If multiple candidate items fit, prefer the one(s) that match the user's intent best.

Context:
{context}

User request:
{question}

Output Format (JSON):
{{
  "recommendation": "<short recommendation title>",
  "explanation": "<short explanation referencing context entries>",
  "used_items": ["list of context items you used"]
}}

Be concise, factual, and base everything only on the context provided.
"""

prompt_template = PromptTemplate.from_template(BASE_PROMPT)

####################################################################
# File: app/services/llm_service.py
####################################################################
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import HumanMessage
from app.services.vector_store import VectorStore
from app.services.prompts import prompt_template
from app.config import settings
import json

class LLMService:
    def __init__(self):
        self.vs = VectorStore()
        # Minimal LLM wrapper — using OpenAI Chat model via LangChain for demo.
        # Swap this with a Gemini client or other provider by editing this block.
        if settings.llm_provider.lower() == "openai":
            # ChatOpenAI is convenient for chat-style models
            self.llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature, openai_api_key=settings.openai_api_key)
        else:
            # If you add other providers, encapsulate them here.
            raise NotImplementedError("Only OpenAI provider demo is included. Add Gemini or others as needed.")

    def recommend(self, question: str, k: int = 4) -> dict:
        # 1) retrieve top-k documents from vector store
        docs = self.vs.get_top_documents(query=question, k=k)
        context = "\n\n".join(docs) if docs else ""

        # 2) build prompt and call LLM
        prompt = prompt_template.format(context=context, question=question)

        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        resp = chain.run({'context': context, 'question': question})

        # Attempt to parse JSON-like output (we asked for JSON)
        parsed = None
        try:
            # The model should output a JSON block — try to extract and parse
            import re
            m = re.search(r"\{[\s\S]*\}", resp)
            if m:
                parsed = json.loads(m.group(0))
            else:
                parsed = {"recommendation": resp.strip(), "explanation": "", "used_items": docs}
        except Exception:
            parsed = {"recommendation": resp.strip(), "explanation": "Could not parse model JSON output.", "used_items": docs}

        return {
            "recommendation": parsed.get("recommendation"),
            "explanation": parsed.get("explanation"),
            "retrieved": parsed.get("used_items", docs)
        }


####################################################################
# File: app/api.py
####################################################################
from fastapi import APIRouter, HTTPException
from app.models import RecommendRequest, RecommendResponse, Recommendation
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        out = llm_service.recommend(question=req.question, k=req.k)
        rec = Recommendation(recommendation=out['recommendation'], explanation=out['explanation'], retrieved=out['retrieved'])
        return RecommendResponse(success=True, result=rec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

####################################################################
# File: app/main.py
####################################################################
from fastapi import FastAPI
from app.api import router

app = FastAPI(title="LLM Recommender", version="0.1")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}

####################################################################
# File: scripts/ingest_demo.py
####################################################################
"""
Simple script to ingest a small demo dataset for the vector store.
Run: python scripts/ingest_demo.py
"""
from app.services.vector_store import VectorStore

if __name__ == "__main__":
    vs = VectorStore()
    documents = [
        "Tesla Model X — electric, long-range, comfortable for inter-state travel.",
        "Lamborghini Aventador — high-performance luxury sports car, not optimized for long comfort trips.",
        "Bentley Continental — top-tier luxury, excellent long-distance comfort and amenities.",
        "Range Rover Vogue — luxury SUV, good for long journeys and varied road conditions.",
        "Economy Bus Service — low-cost inter-state bus service, not a company product but example.",
        "Private Jet Charter — fastest but most expensive, suitable for high-budget travel needs.",
    ]

    vs.from_texts(texts=documents)
    print("Ingested demo documents into Chroma at:", vs.store.persist_directory)

####################################################################
# File: Dockerfile  (optional)
####################################################################
# FROM python:3.11-slim
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# ENV PYTHONUNBUFFERED=1
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

####################################################################
# Notes and extension pointers (keep short):
# - To support Gemini: implement a small LLM adapter in app/services/llm_service.py
#   using google-generativeai SDK and replace ChatOpenAI usage.
# - To support remote vector DBs: swap VectorStore wrapper to Pinecone/Qdrant/Weaviate.
# - Add authorization, request logging, and rate-limiting for production.
# - Expand ingest scripts to take CSV/JSON input and metadata for advanced filtering.

# End of project content.
# personalized_AI_Recommendation
