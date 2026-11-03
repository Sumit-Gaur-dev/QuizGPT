"""
Google Gemini Embedding Provider (Future Extensible Provider).
Uses LangChain's Google Generative AI integration.
Does not make API calls or require keys unless explicitly selected.
"""

import os
from langchain_core.embeddings import Embeddings

_GEMINI_MODEL_CACHE: dict[str, Embeddings] = {}


def get_gemini_embeddings(model_name: str = "text-embedding-004") -> Embeddings:
    """
    Initialize Gemini embeddings using langchain-google-genai.
    Only called when EMBEDDING_PROVIDER=gemini.
    
    Args:
        model_name: Gemini embedding model name (default: text-embedding-004).
        
    Returns:
        LangChain-compatible Embeddings instance.
    """
    if model_name in _GEMINI_MODEL_CACHE:
        return _GEMINI_MODEL_CACHE[model_name]

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip() or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your .env file "
            "or switch back to EMBEDDING_PROVIDER=huggingface to run locally without an API key."
        )

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Gemini embeddings require 'langchain-google-genai'. "
            "Please ensure it is installed in your environment."
        ) from exc

    # Ensure model name format (some versions expect 'models/text-embedding-004')
    formatted_model = model_name if model_name.startswith("models/") else f"models/{model_name}"

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=formatted_model,
            google_api_key=api_key
        )
        _GEMINI_MODEL_CACHE[model_name] = embeddings
        return embeddings
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Gemini embedding model '{model_name}'. "
            f"Error details: {str(exc)}"
        ) from exc

