"""
Text Chunking Module for Quiz RAG using LangChain.
Converts Phase 2 extracted page records into LangChain Document objects
and splits them into overlapping chunks using RecursiveCharacterTextSplitter.
Strictly decoupled from Streamlit UI logic.
"""

import re
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def sanitize_filename_for_id(filename: str) -> str:
    """
    Sanitize a filename for use in a stable, unique chunk identifier.
    Replaces non-alphanumeric characters with underscores.
    """
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return clean_name or "doc"


def convert_extracted_pages_to_documents(doc_record: dict[str, Any]) -> list[Document]:
    """
    Convert an extraction record from rag/ingestion.py into LangChain Document objects.
    Each extracted page becomes a distinct Document with source and page metadata.
    
    Args:
        doc_record: Dictionary returned by extract_text_from_pdf containing:
                    - 'filename': str
                    - 'pages': list of {'page_number': int, 'text': str, ...}
                    
    Returns:
        List of LangChain Document objects for non-empty pages.
    """
    if not doc_record.get("success", False):
        return []

    filename = doc_record.get("filename", "unknown.pdf")
    pages = doc_record.get("pages", [])
    documents: list[Document] = []

    for page_data in pages:
        text = page_data.get("text", "").strip()
        page_num = page_data.get("page_number", 1)

        # Do not create Document objects for empty page content
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": page_num
            }
        )
        documents.append(doc)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS
) -> list[Document]:
    """
    Split a list of LangChain Document objects into overlapping chunk Documents
    using LangChain's RecursiveCharacterTextSplitter.
    
    NOTE: All chunk sizes and overlaps are measured in CHARACTERS (not tokens).
    
    Args:
        documents: List of parent page Document objects.
        chunk_size: Maximum characters per chunk (default: 1000).
        chunk_overlap: Overlapping characters between consecutive chunks (default: 150).
        
    Returns:
        List of chunk Document objects with stable chunk_id and provenance metadata.
    """
    if not documents:
        return []

    # Initialize LangChain RecursiveCharacterTextSplitter
    # Uses default separators: ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    raw_chunks = text_splitter.split_documents(documents)

    # Filter out any accidental empty chunks and attach stable identifiers
    final_chunks: list[Document] = []
    page_chunk_counters: dict[tuple[str, int], int] = {}

    for chunk in raw_chunks:
        content = chunk.page_content.strip()
        if not content:
            continue

        source = chunk.metadata.get("source", "document.pdf")
        page = chunk.metadata.get("page", 1)

        # Track sequential chunk index per (source, page)
        key = (source, page)
        chunk_idx = page_chunk_counters.get(key, 0) + 1
        page_chunk_counters[key] = chunk_idx

        clean_file_id = sanitize_filename_for_id(source)
        stable_chunk_id = f"{clean_file_id}_p{page}_c{chunk_idx}"

        # Enrich chunk metadata with stable ID and character metrics
        chunk.metadata["chunk_id"] = stable_chunk_id
        chunk.metadata["chunk_index"] = chunk_idx
        chunk.metadata["char_count"] = len(content)

        final_chunks.append(chunk)

    return final_chunks


def chunk_all_extracted_docs(
    extracted_docs: dict[str, dict[str, Any]] | list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS
) -> list[Document]:
    """
    Convert all extracted documents in session into LangChain Documents
    and split them into chunk Documents.
    
    Args:
        extracted_docs: Dict mapping filename -> doc_record, or list of doc_records.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap in characters.
        
    Returns:
        List of all chunk LangChain Document objects across all active documents.
    """
    if isinstance(extracted_docs, dict):
        docs_list = list(extracted_docs.values())
    else:
        docs_list = extracted_docs

    parent_documents: list[Document] = []
    for doc_record in docs_list:
        parent_documents.extend(convert_extracted_pages_to_documents(doc_record))

    return split_documents(parent_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
