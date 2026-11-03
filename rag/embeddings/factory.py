"""
Embedding Provider Factory for Quiz RAG.
Instantiates and caches LangChain Embeddings based on active configuration.
"""

from typing import Any
from langchain_core.embeddings import Embeddings

from config.settings import (
    EMBEDDING_PROVIDER,
    HUGGINGFACE_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL
)
from rag.embeddings.providers.huggingface import get_huggingface_embeddings
from rag.embeddings.providers.gemini import get_gemini_embeddings
from rag.embeddings.providers.openai import get_openai_embeddings

SUPPORTED_PROVIDERS = ("huggingface", "gemini", "openai")


def get_embedding_model(
    provider_name: str | None = None,
    model_name: str | None = None
) -> Embeddings:
    """
    Factory function to retrieve or initialize the configured embedding model.
    Conforms strictly to LangChain's Embeddings interface.
    
    Args:
        provider_name: 'huggingface', 'gemini', or 'openai' (defaults to settings).
        model_name: Specific model identifier override (defaults to settings per provider).
        
    Returns:
        Embeddings instance.
        
    Raises:
        ValueError: If provider is unsupported or required credentials are missing.
    """
    provider = (provider_name or EMBEDDING_PROVIDER).lower().strip()

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    if provider == "huggingface":
        resolved_model = model_name or HUGGINGFACE_EMBEDDING_MODEL
        return get_huggingface_embeddings(model_name=resolved_model)

    elif provider == "gemini":
        resolved_model = model_name or GEMINI_EMBEDDING_MODEL
        return get_gemini_embeddings(model_name=resolved_model)

    elif provider == "openai":
        resolved_model = model_name or OPENAI_EMBEDDING_MODEL
        return get_openai_embeddings(model_name=resolved_model)

    raise ValueError(f"Provider '{provider}' is not implemented.")


def get_active_provider_info(
    provider_name: str | None = None,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Return clean status metadata about the active provider for UI rendering.
    """
    provider = (provider_name or EMBEDDING_PROVIDER).lower().strip()
    
    if provider == "huggingface":
        model = model_name or HUGGINGFACE_EMBEDDING_MODEL
        display_name = "Hugging Face (Local)"
        is_local = True
    elif provider == "gemini":
        model = model_name or GEMINI_EMBEDDING_MODEL
        display_name = "Google Gemini"
        is_local = False
    elif provider == "openai":
        model = model_name or OPENAI_EMBEDDING_MODEL
        display_name = "OpenAI"
        is_local = False
    else:
        model = "Unknown"
        display_name = provider.capitalize()
        is_local = False

    return {
        "provider": provider,
        "display_name": display_name,
        "model": model,
        "is_local": is_local
    }

