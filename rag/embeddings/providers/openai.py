"""
OpenAI Embedding Provider (Future Extensible Provider).
Uses LangChain's OpenAI integration.
Does not make API calls or require keys unless explicitly selected.
"""

import os
from langchain_core.embeddings import Embeddings

_OPENAI_MODEL_CACHE: dict[str, Embeddings] = {}


def get_openai_embeddings(model_name: str = "text-embedding-3-small") -> Embeddings:
    """
    Initialize OpenAI embeddings using langchain-openai.
    Only called when EMBEDDING_PROVIDER=openai.
    
    Args:
        model_name: OpenAI embedding model name (default: text-embedding-3-small).
        
    Returns:
        LangChain-compatible Embeddings instance.
    """
    if model_name in _OPENAI_MODEL_CACHE:
        return _OPENAI_MODEL_CACHE[model_name]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "OpenAI API key is missing. Set OPENAI_API_KEY in your .env file "
            "or switch back to EMBEDDING_PROVIDER=huggingface to run locally without an API key."
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "OpenAI embeddings require 'langchain-openai'. "
            "Please ensure it is installed in your environment."
        ) from exc

    try:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        _OPENAI_MODEL_CACHE[model_name] = embeddings
        return embeddings
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize OpenAI embedding model '{model_name}'. "
            f"Error details: {str(exc)}"
        ) from exc

