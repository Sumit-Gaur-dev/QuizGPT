"""
ChromaDB Vector Store Management for Quiz RAG.
Uses LangChain Chroma integration with local persistence and session/model isolation.
Never initializes embedding models directly—always obtains them via get_embedding_model().
"""

import re
from typing import Any
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_PERSIST_DIR
from rag.embeddings.factory import get_embedding_model, get_active_provider_info


def get_safe_collection_name(session_id: str, provider: str, model_name: str) -> str:
    """
    Generate a valid, deterministic ChromaDB collection name isolated by:
    - Session ID (visitor boundary)
    - Embedding Provider
    - Embedding Model Name
    
    This guarantees that different embedding models (which have different vector dimensions)
    never share or collide within the same collection.
    
    ChromaDB requirements: 3-63 chars, alphanumeric, underscores, hyphens,
    starting and ending with alphanumeric characters.
    """
    clean_session = re.sub(r"[^a-zA-Z0-9]", "_", session_id).strip("_")[:20]
    clean_provider = re.sub(r"[^a-zA-Z0-9]", "", provider)[:10]
    
    # Extract short model name from paths like 'sentence-transformers/all-MiniLM-L6-v2'
    raw_model = model_name.split("/")[-1]
    clean_model = re.sub(r"[^a-zA-Z0-9]", "_", raw_model).strip("_")[:18]
    
    name = f"{clean_session}_{clean_provider}_{clean_model}".strip("_")
    # Enforce maximum length of 63 characters
    return name[:63].strip("_")


def get_chroma_client() -> chromadb.PersistentClient:
    """Get the local persistent ChromaDB client."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_vector_store(
    session_id: str,
    provider_name: str | None = None,
    model_name: str | None = None
) -> Chroma:
    """
    Obtain a LangChain Chroma vector store instance bound to the active embedding model
    and isolated by session and provider/model.
    
    Args:
        session_id: Active visitor session identifier.
        provider_name: Optional override for embedding provider.
        model_name: Optional override for embedding model.
        
    Returns:
        Chroma vector store instance.
    """
    # 1. Obtain embedding model strictly via the factory
    embedding_model = get_embedding_model(provider_name=provider_name, model_name=model_name)
    
    # 2. Get provider info to build isolated collection name
    info = get_active_provider_info(provider_name=provider_name, model_name=model_name)
    collection_name = get_safe_collection_name(
        session_id=session_id,
        provider=info["provider"],
        model_name=info["model"]
    )

    client = get_chroma_client()

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )


def index_chunks(
    session_id: str,
    documents: list[Document],
    provider_name: str | None = None,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Index a list of chunk Documents into the session's isolated ChromaDB collection.
    Prevents duplicate insertions by checking chunk_id in metadata.
    
    Args:
        session_id: Visitor session identifier.
        documents: List of chunk Document objects from Phase 3.
        provider_name: Optional provider override.
        model_name: Optional model override.
        
    Returns:
        Dictionary with indexing statistics:
        {
            "added": int,
            "skipped": int,
            "total_in_collection": int,
            "collection_name": str,
            "provider": str,
            "model": str
        }
    """
    info = get_active_provider_info(provider_name=provider_name, model_name=model_name)
    collection_name = get_safe_collection_name(
        session_id=session_id,
        provider=info["provider"],
        model_name=info["model"]
    )

    if not documents:
        stats = get_collection_stats(session_id, provider_name, model_name)
        return {
            "added": 0,
            "skipped": 0,
            "total_in_collection": stats["total_chunks"],
            "collection_name": collection_name,
            "provider": info["provider"],
            "model": info["model"]
        }

    vector_store = get_vector_store(
        session_id=session_id,
        provider_name=provider_name,
        model_name=model_name
    )

    # Extract chunk IDs from metadata
    chunk_ids = [
        doc.metadata.get("chunk_id", f"chunk_{idx}")
        for idx, doc in enumerate(documents)
    ]

    # Duplicate Prevention: query existing IDs in this collection
    existing_result = vector_store.get(ids=chunk_ids)
    existing_ids = set(existing_result.get("ids", []))

    new_docs = []
    new_ids = []
    for doc, cid in zip(documents, chunk_ids):
        if cid not in existing_ids:
            new_docs.append(doc)
            new_ids.append(cid)

    # Insert only non-duplicate chunks
    if new_docs:
        vector_store.add_documents(documents=new_docs, ids=new_ids)

    # Retrieve updated collection total
    all_current = vector_store.get()
    total_count = len(all_current.get("ids", []))

    return {
        "added": len(new_docs),
        "skipped": len(existing_ids),
        "total_in_collection": total_count,
        "collection_name": collection_name,
        "provider": info["provider"],
        "model": info["model"]
    }


def get_collection_stats(
    session_id: str,
    provider_name: str | None = None,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Get current chunk counts and metadata for the session's collection.
    """
    info = get_active_provider_info(provider_name=provider_name, model_name=model_name)
    collection_name = get_safe_collection_name(
        session_id=session_id,
        provider=info["provider"],
        model_name=info["model"]
    )

    client = get_chroma_client()
    try:
        col = client.get_collection(name=collection_name)
        count = col.count()
    except Exception:
        count = 0

    return {
        "total_chunks": count,
        "collection_name": collection_name,
        "provider": info["provider"],
        "display_name": info["display_name"],
        "model": info["model"]
    }


def delete_session_vector_store(session_id: str) -> int:
    """
    Delete all ChromaDB collections associated with this session.
    Part of the 'Delete My Data' lifecycle.
    
    Returns:
        Number of collections deleted.
    """
    client = get_chroma_client()
    clean_session = re.sub(r"[^a-zA-Z0-9]", "_", session_id).strip("_")[:20]
    
    deleted_count = 0
    try:
        collections = client.list_collections()
        for col in collections:
            col_name = col.name if hasattr(col, "name") else str(col)
            if col_name.startswith(clean_session):
                client.delete_collection(name=col_name)
                deleted_count += 1
    except Exception:
        pass

    return deleted_count

