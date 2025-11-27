# app/config.py

import os
from dotenv import load_dotenv
from pydantic import Field
# --- FIX 1: Corrected Pydantic V2 Import ---
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables (or .env file).
    """
    # LLM provider: "gemini", "openai", "claude", etc.
    llm_provider: str = "gemini" 

    # API Keys
    # NOTE: Set your GOOGLE_API_KEY in your .env file
    # We use Field(default=...) for a cleaner Pydantic V2 style
    google_api_key: str | None = Field(
        default=os.environ.get("GOOGLE_API_KEY"),
        description="Your Google API Key"
    )
    openai_api_key: str | None = None

    # Vector DB directory
    chroma_dir: str = "./chroma_db"

    # --- FIX 2 & 3: Model and Embedding Updates ---
    llm_model: str = os.environ.get("LLM_MODEL", "gemini-2.5-flash") # Updated model
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-004") # Updated embedding model
    
    # Model tuning
    temperature: float = 0.0

    # --- FIX 1: Pydantic V2 Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore any extra fields in .env or environment
    )

# Instantiate settings for use throughout the application
settings = Settings()

# Set the environment variable for LangChain's auto-detection
os.environ["GOOGLE_API_KEY"] = settings.google_api_key or ""














# from dotenv import load_dotenv
# import os
# from pydantic_settings import BaseSettings
# load_dotenv()

# class Settings(BaseSettings):
#     # LLM provider: "gemini", "openai", "claude", etc.
#     llm_provider: str = "gemini" 

#     # API Keys
#     google_api_key: str | None = os.environ.get("GOOGLE_API_KEY", "AIzaSyCE6t02ui-P5TTd49qvsgT3He1fPUeHuJw")
#     openai_api_key: str | None = None

#     # Vector DB directory
#     chroma_dir: str = "./chroma_db"

#     # Model + tuning
#     llm_model: str = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
#     temperature: float = 0.0

#     class Config:
#         env_file = ".env"


# settings = Settings()
