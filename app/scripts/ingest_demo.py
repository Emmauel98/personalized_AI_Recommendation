# app/scripts/ingest_demo.py

from app.services.vector_store import VectorStore
from app.demo_data import COMPANY_SERVICES

if __name__ == "__main__":
    print("Ingesting service dataset into Chroma...")

    vs = VectorStore()
    vs.ingest_services(COMPANY_SERVICES)

    print("Done. Vector database updated.")
