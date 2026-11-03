"""
Semantic Search and Document Retrieval Module for Quiz RAG.
Retrieves the Top-K most relevant LangChain Document chunks from ChromaDB.
Strictly decoupled from Streamlit UI logic.
"""

from typing import Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import RETRIEVAL_TOP_K
from rag.vector_store import get_vector_store, get_collection_stats


def get_retriever(
    session_id: str,
    top_k: int = RETRIEVAL_TOP_K,
    provider_name: str | None = None,
    model_name: str | None = None
) -> VectorStoreRetriever:
    """
    Create a LangChain VectorStoreRetriever for the session's isolated ChromaDB collection.
    
    Args:
        session_id: Visitor session identifier.
        top_k: Number of most similar chunks to retrieve.
        provider_name: Optional override for embedding provider.
        model_name: Optional override for embedding model.
        
    Returns:
        VectorStoreRetriever configured for similarity search.
    """
    vector_store = get_vector_store(
        session_id=session_id,
        provider_name=provider_name,
        model_name=model_name
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


def retrieve_relevant_chunks(
    session_id: str,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    provider_name: str | None = None,
    model_name: str | None = None
) -> list[Document]:
    """
    Retrieve top-K LangChain Document chunks matching a query using the retriever.
    
    Args:
        session_id: Visitor session identifier.
        query: Natural language query string.
        top_k: Number of top chunks to retrieve.
        provider_name: Optional provider override.
        model_name: Optional model override.
        
    Returns:
        List of relevant LangChain Document objects.
    """
    if not query or not query.strip():
        return []

    # Check if the session collection has any stored documents
    stats = get_collection_stats(session_id, provider_name, model_name)
    if stats.get("total_chunks", 0) == 0:
        return []

    try:
        retriever = get_retriever(
            session_id=session_id,
            top_k=top_k,
            provider_name=provider_name,
            model_name=model_name
        )
        return retriever.invoke(query.strip())
    except Exception:
        return []


def retrieve_chunks_with_scores(
    session_id: str,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    provider_name: str | None = None,
    model_name: str | None = None
) -> list[dict[str, Any]]:
    """
    Retrieve top-K chunks along with their ChromaDB distance values.
    
    IMPORTANT: ChromaDB distance measures geometric separation in vector space.
    Lower distance indicates closer, more similar vectors. It is NOT a percentage.
    
    Args:
        session_id: Visitor session identifier.
        query: Natural language query string.
        top_k: Maximum chunks to retrieve.
        provider_name: Optional provider override.
        model_name: Optional model override.
        
    Returns:
        List of structured result dictionaries:
        [
            {
                "document": Document,
                "chunk_id": str,
                "source": str,
                "page": int,
                "text": str,
                "distance": float,
                "char_count": int,
                "metadata": dict
            }, ...
        ]
    """
    if not query or not query.strip():
        return []

    # Check collection readiness
    stats = get_collection_stats(session_id, provider_name, model_name)
    if stats.get("total_chunks", 0) == 0:
        return []

    try:
        vector_store = get_vector_store(
            session_id=session_id,
            provider_name=provider_name,
            model_name=model_name
        )

        results = vector_store.similarity_search_with_score(
            query=query.strip(),
            k=top_k
        )

        formatted_results: list[dict[str, Any]] = []
        for doc, score in results:
            formatted_results.append({
                "document": doc,
                "chunk_id": doc.metadata.get("chunk_id", "unknown_chunk"),
                "source": doc.metadata.get("source", "Unknown Document"),
                "page": doc.metadata.get("page", 1),
                "text": doc.page_content,
                "distance": round(float(score), 4),
                "char_count": len(doc.page_content),
                "metadata": doc.metadata
            })

        return formatted_results
    except Exception:
        return []

