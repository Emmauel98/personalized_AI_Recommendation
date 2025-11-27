#!/usr/bin/env python3
"""
LLM-Based Recommendation System Demo (refactored)
- Uses LangChain + Gemini + ChromaDB for context-aware recommendations
- Secure: reads API key from environment variable
- Persistent: Chroma persist directory is configurable
"""
from dotenv import load_dotenv
import os
import sys
from typing import List, Dict, Optional

# Replace imports below with the exact package names you use in your environment.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
load_dotenv()
# -------------------------
# Configuration
# -------------------------
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_store")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "company_services")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
TOP_K = int(os.environ.get("RETRIEVER_K", "4"))

if not GEMINI_API_KEY:
    print("Missing GOOGLE_API_KEY environment variable. Set it and re-run.")
    sys.exit(1)

# -------------------------
# Company catalog (example)
# -------------------------
COMPANY_SERVICES = [
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Tesla Model S",
        "description": "Premium electric sedan with 400+ miles range, autopilot, 0-60 in 3.1s",
        "price": "$89,990",
        "best_for": "Long-distance eco-friendly travel, tech enthusiasts",
        "features": "Electric, autonomous driving, minimal maintenance"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Lamborghini Huracán",
        "description": "Italian supercar with V10 engine, top speed 202 mph",
        "price": "$248,295",
        "best_for": "High-performance enthusiasts, luxury travel",
        "features": "Exotic styling, incredible acceleration, premium materials"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Bentley Continental GT",
        "description": "Luxury grand tourer with handcrafted interior, W12 engine",
        "price": "$230,000",
        "best_for": "Comfortable long-distance luxury travel",
        "features": "Ultimate comfort, prestigious brand, powerful yet refined"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Range Rover Autobiography",
        "description": "Premium SUV with off-road capability and luxury interior",
        "price": "$147,000",
        "best_for": "All-terrain travel, families, weather versatility",
        "features": "All-wheel drive, spacious, handles any road condition"
    }
]

# -------------------------
# Helpers: build documents
# -------------------------
def service_to_document(service: Dict) -> Document:
    # Create a compact, embedding-friendly content string.
    content = (
        f"{service['item']}. {service['description']}. "
        f"Best for: {service['best_for']}. Features: {service['features']}. "
        f"Price: {service['price']}"
    )
    # Only include safe metadata fields (avoid leaking internals)
    metadata = {
        "item": service["item"],
        "service_type": service.get("service_type"),
        "price": service.get("price")
    }
    
    print("Content", content)
    print("Metadata", metadata)
    print(Document(page_content=content, metadata=metadata))
    
    return Document(page_content=content, metadata=metadata)

# -------------------------
# Vector store creation (idempotent)
# -------------------------
def create_or_get_vectorstore(services: List[Dict], persist_dir: Optional[str] = PERSIST_DIR) -> Chroma:
    """
    Creates a Chroma collection if it doesn't exist; otherwise returns the existing collection.
    This avoids re-embedding and duplicate vectors across runs.
    """
    # Create embeddings client
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    # Build documents from services
    documents = [service_to_document(s) for s in services]

    # Chroma.from_documents will reuse the collection when using same persist_directory & collection_name.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir
    )

    # persist to disk (Chroma's API may require explicit persist call depending on version)
    try:
        vectorstore.persist()
    except Exception:
        # some versions persist automatically; ignore if not available
        pass

    return vectorstore

# -------------------------
# Recommendation system
# -------------------------
class RecommendationSystem:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE
        )

        # Create a clearer prompt structure (system + human + context)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert recommendation assistant for a luxury vehicle dealership.\n\n"
                "Only recommend from the following inventory: Tesla Model S, Lamborghini Huracán, "
                "Bentley Continental GT, Range Rover Autobiography.\n\n"
                "When answering, provide:\n"
                "1) The single BEST vehicle recommendation from our inventory.\n"
                "2) Concise reasoning why it fits the user's explicit needs.\n"
                "3) One or two alternative options from the inventory (if applicable) and why.\n"
                "4) Key benefits that address their travel requirements.\n\n"
                "Use the context provided (catalog snippets) to ground your answer. Be persuasive but honest."
            )),
            ("human", "Context:\n{context}\n\nUser request:\n{question}")
        ])

    def get_recommendation(self, user_request: str) -> Dict:
        # Use retriever to get relevant docs
        relevant_docs = self.retriever.invoke(user_request)

        # Build context (join doc contents)
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

        # Format messages for the LLM
        messages = self.prompt_template.format_messages(context=context, question=user_request)

        # Invoke the LLM; keep robust error handling
        try:
            response = self.llm.invoke(messages)
            text = getattr(response, "content", None) or str(response)
        except Exception as e:
            text = f"Error calling LLM: {e}"

        # Return safe metadata for retrieved services
        retrieved_safe = [doc.metadata for doc in relevant_docs]

        return {
            "recommendation": text,
            "relevant_services": retrieved_safe
        }

# -------------------------
# Demo runner / CLI
# -------------------------
def run_demo(interactive: bool = True):
    # print("LUXURY VEHICLE RECOMMENDATION SYSTEM (Refactored)")
    print("AIRECOMMENDATION SYSTEM")
    print("Initializing vector store and LLM...")

    vectorstore = create_or_get_vectorstore(COMPANY_SERVICES, persist_dir=PERSIST_DIR)
    print(f"Vector store ready (collection='{COLLECTION_NAME}', persist_dir='{PERSIST_DIR}')")

    rec_system = RecommendationSystem(vectorstore)
    print(f"Recommendation system initialized (llm_model={LLM_MODEL}, temp={LLM_TEMPERATURE})")

    # A few demo queries to show output
    demo_queries = [
        "I need to travel from California to New York. What do you recommend?",
        "I'm looking for something fast and luxurious for weekend trips",
        "I need a reliable vehicle for family trips in various weather conditions",
        "What's the most environmentally friendly option for long-distance travel?"
    ]

    for i, q in enumerate(demo_queries, 1):
        print("\n" + "-" * 60)
        print(f"DEMO QUERY {i}: {q}")
        res = rec_system.get_recommendation(q)
        print("\nAI RECOMMENDATION:\n")
        print(res["recommendation"])
        print("\nRetrieved Services:")
        for s in res["relevant_services"]:
            print(f"  • {s.get('item')} — {s.get('price')}")
    print("\n" + "=" * 70)

    if interactive:
        print("\nInteractive mode (type 'quit' to exit).")
        while True:
            try:
                user_input = input("\nYour request: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            result = rec_system.get_recommendation(user_input)
            print("\n" + "-" * 40)
            print(result["recommendation"])
            print("-" * 40)

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        run_demo(interactive=True)
    except Exception as ex:
        print(f"Fatal error: {ex}")
        raise
#!/usr/bin/env python3
"""
LLM-Based Recommendation System Demo (Refactored & Corrected)
- Uses LangChain + Gemini + ChromaDB
- Correct Google model names (LLM vs Embeddings)
- Secure: dotenv + environment variables
- Idempotent vectorstore creation
"""

from dotenv import load_dotenv
import os
import sys
from typing import List, Dict, Optional

# LangChain & Gemini imports
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load .env
load_dotenv()

# -------------------------
# Environment Variables
# -------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "company_services")

# IMPORTANT: Correct model names
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")  # NO "models/"
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-1.5-flash")  # With "models/"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
TOP_K = int(os.getenv("RETRIEVER_K", "4"))

if not GEMINI_API_KEY:
    print("ERROR: Missing GOOGLE_API_KEY in environment variables.")
    sys.exit(1)

# -------------------------
# Example Company Services
# -------------------------
COMPANY_SERVICES = [
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Tesla Model S",
        "description": "Premium electric sedan with 400+ miles range, autopilot, 0-60 in 3.1s",
        "price": "$89,990",
        "best_for": "Long-distance eco-friendly travel, tech enthusiasts",
        "features": "Electric, autonomous driving, minimal maintenance"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Lamborghini Huracán",
        "description": "Italian supercar with V10 engine, top speed 202 mph",
        "price": "$248,295",
        "best_for": "High-performance enthusiasts, luxury travel",
        "features": "Exotic styling, incredible acceleration, premium materials"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Bentley Continental GT",
        "description": "Luxury grand tourer with handcrafted interior, W12 engine",
        "price": "$230,000",
        "best_for": "Comfortable long-distance luxury travel",
        "features": "Ultimate comfort, prestigious brand, powerful yet refined"
    },
    {
        "service_type": "Luxury Vehicle Sales",
        "item": "Range Rover Autobiography",
        "description": "Premium SUV with off-road capability and luxury interior",
        "price": "$147,000",
        "best_for": "All-terrain travel, families, weather versatility",
        "features": "All-wheel drive, spacious, handles any road condition"
    }
]

# -------------------------
# Convert Service → Document
# -------------------------
def service_to_document(service: Dict) -> Document:
    content = (
        f"{service['item']}. {service['description']}. "
        f"Best for: {service['best_for']}. Features: {service['features']}. "
        f"Price: {service['price']}"
    )

    metadata = {
        "item": service["item"],
        "service_type": service.get("service_type"),
        "price": service.get("price")
    }

    return Document(page_content=content, metadata=metadata)

# -------------------------
# ChromaDB Initialization
# -------------------------
def create_or_get_vectorstore(
    services: List[Dict],
    persist_dir: Optional[str] = PERSIST_DIR
) -> Chroma:

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,  # ✔ correct model format
        google_api_key=GEMINI_API_KEY
    )

    documents = [service_to_document(s) for s in services]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir
    )

    try:
        vectorstore.persist()
    except Exception:
        pass

    return vectorstore

# -------------------------
# Recommendation Engine
# -------------------------
class RecommendationSystem:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,    # ✔ correct name
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert luxury vehicle recommendation assistant.\n"
                "Only recommend from this inventory: Tesla Model S, Lamborghini Huracán, "
                "Bentley Continental GT, Range Rover Autobiography.\n\n"
                "Provide:\n"
                "1) Best single recommendation\n"
                "2) Short reasoning\n"
                "3) 1–2 alternative options\n"
                "4) Key benefits for the user's needs"
            ),
            ("human", "Context:\n{context}\n\nUser request:\n{question}")
        ])

    def get_recommendation(self, user_request: str) -> Dict:
        docs = self.retriever.invoke(user_request)
        context = "\n\n".join(d.page_content for d in docs)

        messages = self.prompt.format_messages(
            context=context,
            question=user_request
        )

        try:
            response = self.llm.invoke(messages)
            text = response.content
        except Exception as e:
            text = f"ERROR calling LLM: {e}"

        return {
            "recommendation": text,
            "relevant_services": [d.metadata for d in docs]
        }

# -------------------------
# CLI Demo Runner
# -------------------------
def run_demo(interactive: bool = True):
    print("\nInitializing Vector Store + Gemini Models...\n")

    vectorstore = create_or_get_vectorstore(COMPANY_SERVICES)
    rec = RecommendationSystem(vectorstore)

    sample_queries = [
        "I need to travel from California to New York. What do you recommend?",
        "I want something very fast and luxurious",
        "I need a reliable family car for all weather",
        "What is the most eco-friendly long-distance car?"
    ]

    for i, q in enumerate(sample_queries, 1):
        print(f"\n--- DEMO QUERY {i}: {q} ---\n")
        result = rec.get_recommendation(q)
        print(result["recommendation"])
        print("\nRelevant services:")
        for s in result["relevant_services"]:
            print(" •", s)

    if not interactive:
        return

    print("\nInteractive Mode (type 'quit' to exit)\n")
    while True:
        try:
            user_q = input("Your request: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        if user_q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return

        result = rec.get_recommendation(user_q)
        print("\n", result["recommendation"], "\n")

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    run_demo(interactive=True)
