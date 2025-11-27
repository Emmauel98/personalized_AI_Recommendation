"""
LLM-Based Recommendation System Demo
Uses LangChain + Gemini + ChromaDB for context-aware recommendations
"""

import os
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your Gemini API key here or as environment variable
# Get free key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

# ============================================================================
# COMPANY SERVICE DATA
# ============================================================================

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

# ============================================================================
# VECTOR DATABASE SETUP
# ============================================================================

def create_vector_store(services: List[Dict]) -> Chroma:
    """Create a vector database from company services."""
    
    # Convert service data to documents
    documents = []
    for service in services:
        content = f"""
Service: {service['item']}
Type: {service['service_type']}
Description: {service['description']}
Price: {service['price']}
Best For: {service['best_for']}
Features: {service['features']}
        """
        doc = Document(page_content=content, metadata=service)
        documents.append(doc)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="company_services"
    )
    
    return vectorstore

# ============================================================================
# LLM RECOMMENDATION SYSTEM
# ============================================================================

class RecommendationSystem:
    """Intelligent recommendation system using LLM and vector search."""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        
        # Create recommendation prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert recommendation assistant for a luxury vehicle dealership.
            
Your company ONLY sells these vehicles: Tesla Model S, Lamborghini Huracán, Bentley Continental GT, and Range Rover Autobiography.

Based on the customer's request and the available services from our catalog, provide:
1. The BEST vehicle recommendation from our inventory
2. Clear reasoning why this is the best choice for their specific needs
3. Alternative options from our inventory if applicable
4. Key benefits that address their travel requirements

Use the context provided about our vehicles to make informed recommendations.
Be persuasive but honest. If their request seems unusual for buying a car (like a short trip), 
acknowledge it but still recommend the best vehicle for their broader needs.

Context from our catalog:
{context}"""),
            ("human", "{question}")
        ])
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
    
    def get_recommendation(self, user_request: str) -> Dict:
        """Get recommendation for user request."""
        
        # Retrieve relevant services
        relevant_docs = self.vectorstore.similarity_search(user_request, k=4)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create the full prompt
        messages = self.prompt_template.format_messages(
            context=context,
            question=user_request
        )
        
        # Get LLM response
        response = self.llm.invoke(messages)
        
        return {
            "recommendation": response.content,
            "relevant_services": [doc.metadata for doc in relevant_docs]
        }

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def run_demo():
    """Run the recommendation system demo."""
    
    print("=" * 70)
    print("🚗 LUXURY VEHICLE RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nInitializing system...")
    
    # Create vector store
    vectorstore = create_vector_store(COMPANY_SERVICES)
    print("✅ Vector database created with company services")
    
    # Initialize recommendation system
    rec_system = RecommendationSystem(vectorstore)
    print("✅ Recommendation system initialized with Gemini LLM")
    
    # Demo queries
    demo_queries = [
        "I need to travel from California to New York. What do you recommend?",
        "I'm looking for something fast and luxurious for weekend trips",
        "I need a reliable vehicle for family trips in various weather conditions",
        "What's the most environmentally friendly option for long-distance travel?"
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING DEMO SCENARIOS")
    print("=" * 70)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n\n📝 SCENARIO {i}")
        print(f"User Request: {query}")
        print("-" * 70)
        
        result = rec_system.get_recommendation(query)
        
        print("\n🤖 AI RECOMMENDATION:")
        print(result["recommendation"])
        
        print("\n📊 Retrieved Services:")
        for service in result["relevant_services"]:
            print(f"  • {service['item']} - {service['price']}")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)

    # Interactive mode
    print("\n\n🎯 INTERACTIVE MODE")
    print("Enter your travel requirements (or 'quit' to exit):")
    
    while True:
        user_input = input("\nYour request: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using our recommendation system!")
            break
        
        if not user_input:
            continue
        
        print("\n🤖 Processing your request...\n")
        result = rec_system.get_recommendation(user_input)
        print(result["recommendation"])

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check API key
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Please set your GOOGLE_API_KEY!")
        print("Get a free key from: https://makersuite.google.com/app/apikey")
        print("\nSet it as environment variable:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("\nOr edit the script and replace YOUR_API_KEY_HERE")
    else:
        run_demo()