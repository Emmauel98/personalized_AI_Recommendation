# app/services/llm_service.py

from dotenv import load_dotenv
import os
import sys
from typing import List, Dict, Optional

# Replace imports below with the exact package names you use in your environment.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import json
import re
# from langchain_community.chains import LLMChain
# from langchain import PromptTemplate

load_dotenv()

# Optional: allow OpenAI too
try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from app.services.vector_store import VectorStore
from app.config import settings


# -------------------------
# Prompt template
# -------------------------
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an intelligent recommendation assistant.

# Use ONLY the information from the provided context below.
# If the context does not contain enough information, say so.

# ---

# CONTEXT:
# {context}

# ---

# QUESTION:
# {question}

# ---

# Return a JSON object ONLY in this format:

# {{
#   "recommendation": "short answer",
#   "explanation": "why this is recommended",
#   "used_items": ["list of items used"]
# }}
# """
# )


class LLMService:
    """
    Handles:
    - LLM provider selection (Gemini or OpenAI)
    - Retrieval + reasoning
    - JSON-safe recommendation output
    """

    def __init__(self):
        self.vectorStore = VectorStore()

        provider = settings.llm_provider.lower()

        self.llm = self._initialize_llm(provider)

        # Initialize the ChatPromptTemplate here using the JSON_PROMPT_TEMPLATE structure
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert recommendation assistant for a luxury vehicle dealership.\n\n"
             "Only recommend from the provided context (catalog snippets). Your final output "
             "MUST be a JSON object formatted exactly as specified below."
             "Final JSON format: {json_schema}"), # Inject a hint for better JSON reliability
            ("human", "Context:\n{context}\n\nUser request:\n{question}")
        ])
        
        # Prepare the expected JSON structure string for the prompt
        self.json_schema_str = json.dumps({"recommendation": "...", "explanation": "...", "used_items": ["..."]}, indent=2)


    def _initialize_llm(self, provider: str):
        """Helper to initialize the correct LLM based on provider."""
        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.temperature
            )

        elif provider == "openai":
            if ChatOpenAI is None:
                 # Inform the user what to install
                raise ImportError("OpenAI model is not installed. Run 'pip install langchain-openai'.") 
            return ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=settings.openai_api_key,
                temperature=settings.temperature
            )

        else:
            raise NotImplementedError(f"Unknown provider: {provider}")
        
        # if provider == "gemini":
        #     self.llm = ChatGoogleGenerativeAI(
        #         model=settings.llm_model,  # e.g. "models/gemini-1.5-flash"
        #         google_api_key=settings.google_api_key,
        #         temperature=settings.temperature
        #     )

        # elif provider == "openai":
        #     if ChatOpenAI is None:
        #         raise ImportError("OpenAI model is not installed.")
        #     self.llm = ChatOpenAI(
        #         model=settings.llm_model,
        #         openai_api_key=settings.openai_api_key,
        #         temperature=settings.temperature
        #     )

        # else:
        #     raise NotImplementedError(f"Unknown provider: {provider}")

    # ---------------------------------------------------------
    # Main recommendation function
    # ---------------------------------------------------------
    def recommend(self, user_request: str, k: int = 4) -> dict:
        # 1. Retrieve vector documents
        docs = self.vectorStore.get_top_documents(query=user_request, k=k)
        
        # 2. Build context string from retrieved documents
        context_str = "\n\n".join(d.page_content for d in docs)
        
        # 3. Format the chat messages for the LLM
        messages = self.chat_prompt.format_messages(
            context=context_str, 
            question=user_request,
            json_schema=self.json_schema_str
        )
        
        # 4. Invoke the LLM directly
        try:
            response = self.llm.invoke(messages)
            text = response.content
        except Exception as e:
            text = f"Error calling LLM: {e}"
            
        # 5. Extract JSON safely (using the logic you commented out, now re-implemented)
        parsed = self._safe_json_parse(text)

        # 6. Attach metadata from documents
        parsed["retrieved"] = [d.metadata for d in docs]
        
        return parsed

    def _safe_json_parse(self, resp: str) -> dict:
        """Extracts and parses the JSON block from the LLM response."""
        try:
            # Regex to find the first JSON object block
            match = re.search(r"\{[\s\S]*\}", resp)
            if match:
                return json.loads(match.group(0))
            else:
                raise ValueError("No JSON block found.")
        except Exception:
            # Fallback if parsing fails
            return {
                "recommendation": resp.strip(),
                "explanation": "Failed to parse model JSON or JSON block was missing.",
                "used_items": []
            }
        
        # # Use retriever to get relevant docs
        # relevant_docs = self.retriever.invoke(user_request)
        
        # self.prompt_template = ChatPromptTemplate.from_messages([
        #     ("system", (
        #         "You are an expert recommendation assistant for a luxury vehicle dealership.\n\n"
        #         "Only recommend from the following inventory: Tesla Model S, Lamborghini Huracán, "
        #         "Bentley Continental GT, Range Rover Autobiography.\n\n"
        #         "When answering, provide:\n"
        #         "1) The single BEST vehicle recommendation from our inventory.\n"
        #         "2) Concise reasoning why it fits the user's explicit needs.\n"
        #         "3) One or two alternative options from the inventory (if applicable) and why.\n"
        #         "4) Key benefits that address their travel requirements.\n\n"
        #         "Use the context provided (catalog snippets) to ground your answer. Be persuasive but honest."
        #     )),
        #     ("human", "Context:\n{context}\n\nUser request:\n{question}")
        # ])
        
        # # Build context (join doc contents)
        # context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""


        # context_str = "\n\n".join(d.page_content for d in docs)
        
        # messages = self.prompt_template.format_messages(context=context, question=user_request)

        # # Invoke the LLM; keep robust error handling
        # try:
        #     response = self.llm.invoke(messages)
        #     text = getattr(response, "content", None) or str(response)
        # except Exception as e:
        #     text = f"Error calling LLM: {e}"

        # # Return safe metadata for retrieved services
        # retrieved_safe = [doc.metadata for doc in relevadicnt_docs]

        # return {
        #     "recommendation": text,
        #     "relevant_services": retrieved_safe
        # }


        # # 2 Build the chain
        # chain = LLMChain(llm=self.llm, prompt=prompt_template)

        # # 3 Run the chain
        # resp = chain.run({"context": context_str, "question": question})

        # # 4 Extract JSON safely
        # parsed = None
        # try:
        #     match = re.search(r"\{[\s\S]*\}", resp)
        #     if match:
        #         parsed = json.loads(match.group(0))
        #     else:
        #         parsed = {
        #             "recommendation": resp.strip(),
        #             "explanation": "Model did not return JSON format.",
        #             "used_items": []
        #         }
        # except Exception:
        #     parsed = {
        #         "recommendation": resp.strip(),
        #         "explanation": "Failed to parse model JSON.",
        #         "used_items": []
        #     }

        # # 5 Attach metadata from documents
        # parsed["retrieved"] = [d.metadata for d in docs]

        # return parsed
