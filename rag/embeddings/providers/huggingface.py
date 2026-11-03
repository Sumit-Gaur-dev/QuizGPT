"""
Hugging Face Sentence Transformers Embedding Provider (Local V1 Provider).
Generates embeddings locally using sentence-transformers without requiring API keys.
"""

import logging
from typing import Any
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Module-level cache to ensure the model is loaded only once
_HF_MODEL_CACHE: dict[str, Embeddings] = {}


def get_huggingface_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embeddings:
    """
    Get or initialize the Hugging Face Sentence Transformers embedding model.
    Runs locally and caches the model in memory.
    
    Args:
        model_name: Hugging Face model identifier.
        
    Returns:
        LangChain-compatible Embeddings instance.
    """
    if model_name in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[model_name]

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Hugging Face embeddings require 'langchain-huggingface' and 'sentence-transformers'. "
                "Please ensure they are installed in your environment."
            ) from exc

    try:
        logger.info(f"Loading local Hugging Face embedding model: '{model_name}'...")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        _HF_MODEL_CACHE[model_name] = embeddings
        return embeddings
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face embedding model '{model_name}'. "
            f"Error details: {str(exc)}"
        ) from exc

