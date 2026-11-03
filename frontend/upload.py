"""
Upload and document processing view for Quiz RAG.
Handles PDF uploads, validation, text extraction, LangChain chunking,
and modular embedding generation into local ChromaDB.
"""

import streamlit as st
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from config.settings import (
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_SESSION,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS
)
from rag.ingestion import extract_text_from_pdf
from rag.chunking import chunk_all_extracted_docs
from rag.embeddings.factory import get_active_provider_info
from rag.vector_store import index_chunks, get_collection_stats


def generate_sample_pdf_bytes(scanned_mode: bool = False) -> tuple[bytes, str]:
    """
    Generate an in-memory sample PDF for immediate local testing.
    
    Args:
        scanned_mode: If True, generates a page without text to simulate a scanned PDF.
        
    Returns:
        tuple of (pdf_bytes, default_filename).
    """
    doc = fitz.open()
    if scanned_mode:
        doc.new_page(width=595, height=842)
        filename = "Sample_Scanned_Document.pdf"
    else:
        page1 = doc.new_page(width=595, height=842)
        page1.insert_text(
            (50, 72),
            "Lecture 01: Introduction to Retrieval-Augmented Generation (RAG)\n\n"
            "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances "
            "Large Language Models (LLMs) by retrieving authoritative knowledge from an external database "
            "before generating a response.\n\n"
            "Core Architectural Pipeline:\n"
            "1. Ingestion: Extract raw text from source documents (e.g. PDFs) with page numbers.\n"
            "2. Chunking: Divide text into semantically cohesive, overlapping segments using LangChain.\n"
            "3. Embeddings: Convert text chunks into dense high-dimensional vectors.\n"
            "4. Vector Database: Store embeddings with metadata in session collections (e.g. ChromaDB).\n"
            "5. Retrieval: Query the vector space to find Top-K chunks most relevant to a prompt.\n"
            "6. Grounded Generation: Supply retrieved context to the LLM to formulate accurate answers.\n\n"
            "Why RAG is Preferred over Fine-Tuning:\n"
            "- Fine-tuning teaches a model style, tone, or specific syntax, but is inefficient for factual recall.\n"
            "- RAG provides explicit citations, eliminates retraining costs, and updates immediately when new notes are added.",
            fontsize=11
        )

        page2 = doc.new_page(width=595, height=842)
        page2.insert_text(
            (50, 72),
            "Lecture 01 (Continued): Chunking and Vector Embeddings\n\n"
            "Text Chunking Principles:\n"
            "- Chunk Size: Typically 800 to 1,000 characters for tight, focused context windows.\n"
            "- Chunk Overlap: Typically 100 to 150 characters to prevent splitting statements.\n"
            "- Overlap ensures context continuity across split boundaries.\n\n"
            "Vector Space & Similarity:\n"
            "- High-dimensional embedding spaces map semantic meaning into vector proximity.\n"
            "- Cosine similarity or Euclidean distance identifies relevant passages for any question.\n\n"
            "Deterministic vs LLM Evaluation:\n"
            "- Objective questions (MCQs, True/False) must be scored deterministically in code.\n"
            "- Short-answer open responses can be evaluated by an LLM with structured rubric guidance.",
            fontsize=11
        )
        filename = "Lecture_01_RAG_Fundamentals.pdf"

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes, filename


def render_upload() -> None:
    """Render the PDF upload, extraction, LangChain chunking, and ChromaDB vector indexing interface."""
    st.header("📁 Document Ingestion, Chunking & Vector Storage")
    st.markdown(
        "Upload study notes, extract text, split into **LangChain Documents**, "
        "and generate vector embeddings into an isolated **ChromaDB collection**."
    )

    st.divider()

    # Informational limit banners
    col_info1, col_info2, col_info3 = st.columns(3)
    provider_info = get_active_provider_info()
    with col_info1:
        st.caption(f"📏 Max File Size: **{MAX_FILE_SIZE_MB} MB**")
    with col_info2:
        st.caption(f"📑 Max Files: **{MAX_FILES_PER_SESSION} PDFs** per session")
    with col_info3:
        st.caption(f"⚡ Active Provider: **{provider_info['display_name']}**")

    # Upload widget
    uploaded_files = st.file_uploader(
        label="Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based PDF lecture notes or reading materials (V1 supports text PDFs, not OCR)."
    )

    docs_modified = False

    # Process new uploads without reprocessing already cached files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.extracted_docs:
                file_bytes = file.getvalue()
                result = extract_text_from_pdf(file_bytes=file_bytes, filename=file.name)
                
                # Cache the extraction result
                st.session_state.extracted_docs[file.name] = result

                # Register metadata in uploaded_files list
                st.session_state.uploaded_files.append({
                    "id": f"upload_{file.name}",
                    "name": file.name,
                    "size_kb": round(len(file_bytes) / 1024, 1),
                    "total_pages": result["total_pages"],
                    "extracted_pages": result["extracted_pages_count"],
                    "total_characters": result["total_characters"],
                    "status": "Extracted" if result["success"] else "Failed",
                    "warning": result.get("warning"),
                    "error": result.get("error")
                })
                docs_modified = True

    # Quick test sample note loaders
    with st.expander("💡 Quick Test: Load Built-in Sample PDFs", expanded=False):
        st.caption("Test extraction, LangChain chunking, and ChromaDB vector indexing:")
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("📄 Load 2-Page Study Note (Text PDF)", use_container_width=True):
                sample_bytes, sample_name = generate_sample_pdf_bytes(scanned_mode=False)
                if sample_name not in st.session_state.extracted_docs:
                    result = extract_text_from_pdf(sample_bytes, sample_name)
                    st.session_state.extracted_docs[sample_name] = result
                    st.session_state.uploaded_files.append({
                        "id": f"upload_{sample_name}",
                        "name": sample_name,
                        "size_kb": round(len(sample_bytes) / 1024, 1),
                        "total_pages": result["total_pages"],
                        "extracted_pages": result["extracted_pages_count"],
                        "total_characters": result["total_characters"],
                        "status": "Extracted",
                        "warning": None,
                        "error": None
                    })
                    # Recompute chunks with LangChain
                    st.session_state.chunks = chunk_all_extracted_docs(st.session_state.extracted_docs)
                    st.toast("Sample PDF loaded, extracted, and chunked!", icon="✅")
                    st.rerun()
                else:
                    st.info("Sample study note is already loaded.")

        with col_btn2:
            if st.button("🖼️ Test Scanned / Image-Only PDF", use_container_width=True):
                scanned_bytes, scanned_name = generate_sample_pdf_bytes(scanned_mode=True)
                if scanned_name not in st.session_state.extracted_docs:
                    result = extract_text_from_pdf(scanned_bytes, scanned_name)
                    st.session_state.extracted_docs[scanned_name] = result
                    st.session_state.uploaded_files.append({
                        "id": f"upload_{scanned_name}",
                        "name": scanned_name,
                        "size_kb": round(len(scanned_bytes) / 1024, 1),
                        "total_pages": result["total_pages"],
                        "extracted_pages": result["extracted_pages_count"],
                        "total_characters": result["total_characters"],
                        "status": "Failed",
                        "warning": None,
                        "error": result["error"]
                    })
                    st.toast("Scanned PDF validation triggered!", icon="⚠️")
                    st.rerun()
                else:
                    st.info("Scanned document test is already in session.")

    # Automatically recompute LangChain chunks when documents are modified
    if docs_modified:
        st.session_state.chunks = chunk_all_extracted_docs(st.session_state.extracted_docs)

    st.markdown("---")

    # Pipeline Statistics Summary
    extracted_docs = st.session_state.get("extracted_docs", {})
    valid_docs = [d for d in extracted_docs.values() if d.get("success")]
    total_pages = sum(d.get("extracted_pages_count", 0) for d in valid_docs)
    total_chunks = len(st.session_state.get("chunks", []))
    total_chars = sum(c.metadata.get("char_count", len(c.page_content)) for c in st.session_state.get("chunks", []))
    avg_chunk_size = round(total_chars / total_chunks, 1) if total_chunks > 0 else 0
    chroma_stats = get_collection_stats(st.session_state.session_id)

    st.subheader("📊 Pipeline Statistics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Valid Documents", value=len(valid_docs))
    with m2:
        st.metric(label="Extracted Pages", value=total_pages)
    with m3:
        st.metric(label="LangChain Chunks", value=total_chunks)
    with m4:
        st.metric(
            label="ChromaDB Indexed",
            value=chroma_stats["total_chunks"],
            delta="Ready for Retrieval" if chroma_stats["total_chunks"] > 0 else "Needs Indexing"
        )

    st.divider()

    # Section 1: Ingested Documents List
    st.subheader(f"📋 Ingested Documents ({len(st.session_state.uploaded_files)})")

    if not st.session_state.uploaded_files:
        st.info("No documents uploaded yet. Upload a PDF above or load a sample note to inspect extraction.")
    else:
        for idx, file_meta in enumerate(st.session_state.uploaded_files):
            doc_result = st.session_state.extracted_docs.get(file_meta["name"], {})
            is_success = doc_result.get("success", False)

            with st.container(border=True):
                col_icon, col_details, col_meta, col_del = st.columns([0.5, 3.5, 3, 1])

                with col_icon:
                    st.markdown("### 📄" if is_success else "### ⚠️")

                with col_details:
                    st.markdown(f"**{file_meta['name']}**")
                    st.caption(f"Size: `{file_meta['size_kb']} KB` | Pages: `{file_meta['total_pages']}`")

                with col_meta:
                    if is_success:
                        doc_chunks_count = sum(
                            1 for c in st.session_state.chunks
                            if c.metadata.get("source") == file_meta["name"]
                        )
                        st.success(
                            f"Pages: **{file_meta['extracted_pages']}/{file_meta['total_pages']}** | "
                            f"Chunks: **{doc_chunks_count}**"
                        )
                    else:
                        st.error("Extraction Failed")

                with col_del:
                    if st.button("Remove", key=f"remove_doc_{idx}", use_container_width=True):
                        del st.session_state.extracted_docs[file_meta["name"]]
                        st.session_state.uploaded_files.pop(idx)
                        st.session_state.chunks = chunk_all_extracted_docs(st.session_state.extracted_docs)
                        st.rerun()

                if not is_success:
                    st.error(f"❌ **Validation Error:** {doc_result.get('error', 'Unknown extraction error')}")

                if is_success and doc_result.get("warning"):
                    st.warning(f"⚠️ {doc_result['warning']}")

                if is_success and doc_result.get("pages"):
                    with st.expander(f"🔍 Raw Text Preview: {file_meta['name']} ({len(doc_result['pages'])} pages)", expanded=False):
                        for page in doc_result["pages"]:
                            st.markdown(f"**Page {page['page_number']}** `({page['char_count']} characters)`")
                            st.text_area(
                                label=f"Page {page['page_number']} Text",
                                value=page["text"],
                                height=100,
                                disabled=True,
                                key=f"preview_{file_meta['name']}_p{page['page_number']}",
                                label_visibility="collapsed"
                            )

    st.markdown("---")

    # Section 2: LangChain Chunks Inspector
    st.subheader(f"🧩 LangChain Chunks Inspector ({total_chunks} Chunks)")

    if total_chunks == 0:
        st.info("No chunks generated yet. Upload a valid document to inspect LangChain chunking.")
    else:
        with st.expander("⚙️ LangChain Splitter Parameters & Re-chunking", expanded=False):
            st.markdown(
                "Customize the **LangChain RecursiveCharacterTextSplitter** parameters. "
                "All settings are measured in **characters** (approx. 4 characters per token)."
            )
            c_size_col, c_overlap_col = st.columns(2)
            with c_size_col:
                custom_size = st.slider(
                    label="Chunk Size (Characters)",
                    min_value=300,
                    max_value=2500,
                    value=CHUNK_SIZE_CHARS,
                    step=50,
                    help="Maximum character length per chunk."
                )
            with c_overlap_col:
                custom_overlap = st.slider(
                    label="Chunk Overlap (Characters)",
                    min_value=20,
                    max_value=500,
                    value=CHUNK_OVERLAP_CHARS,
                    step=10,
                    help="Target overlapping characters between consecutive chunks."
                )

            if st.button("Apply Parameters & Re-chunk with LangChain", type="secondary"):
                st.session_state.chunks = chunk_all_extracted_docs(
                    st.session_state.extracted_docs,
                    chunk_size=custom_size,
                    chunk_overlap=custom_overlap
                )
                st.toast(f"Re-chunked with size={custom_size}, overlap={custom_overlap}!", icon="🧩")
                st.rerun()

        with st.expander(f"🔍 View All Chunks ({total_chunks}) - LangChain Documents", expanded=False):
            for idx, chunk in enumerate(st.session_state.chunks):
                with st.container(border=True):
                    ch_header, ch_badge = st.columns([3, 1])
                    with ch_header:
                        st.markdown(f"**Chunk #{idx + 1}:** `{chunk.metadata.get('chunk_id', f'chunk_{idx+1}')}`")
                        st.caption(
                            f"📄 Source: `{chunk.metadata.get('source')}` | "
                            f"📑 Page: **{chunk.metadata.get('page')}** | "
                            f"📏 Length: **{chunk.metadata.get('char_count', len(chunk.page_content))} chars**"
                        )
                    with ch_badge:
                        st.info(f"Page {chunk.metadata.get('page')}")

                    st.text_area(
                        label=f"Chunk {idx + 1} Content",
                        value=chunk.page_content,
                        height=90,
                        disabled=True,
                        key=f"chunk_view_{chunk.metadata.get('chunk_id', idx)}_{idx}",
                        label_visibility="collapsed"
                    )

                    with st.expander("Document Metadata", expanded=False):
                        st.json(chunk.metadata)

    st.markdown("---")

    # Section 3: Vector Embeddings & ChromaDB Storage (Phase 4)
    st.subheader("⚡ Vector Embeddings & ChromaDB Storage")
    st.markdown(
        "Generate vector embeddings and index them into an isolated local ChromaDB collection. "
        "Embedding execution is strictly controlled via the button below."
    )

    with st.container(border=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"**Active Provider:** `{provider_info['display_name']}`")
            st.caption("Mode: 100% Local Inference" if provider_info["is_local"] else "Mode: Remote Cloud API")
        with col_p2:
            st.markdown(f"**Model:** `{provider_info['model']}`")
            st.caption(f"Collection: `{chroma_stats['collection_name']}`")
        with col_p3:
            st.markdown(f"**ChromaDB Total:** `{chroma_stats['total_chunks']}` chunks")
            awaiting_count = max(0, total_chunks - chroma_stats["total_chunks"])
            st.caption(f"Unindexed Chunks: **{awaiting_count}**")

        st.write("")

        # Manual generation button
        if st.button("⚡ Generate Embeddings & Store in ChromaDB", type="primary", use_container_width=True):
            if not st.session_state.chunks:
                st.warning("⚠️ No chunks available to index. Upload and chunk study notes first.")
            else:
                with st.spinner(f"Generating embeddings using {provider_info['display_name']} and persisting to ChromaDB..."):
                    try:
                        index_res = index_chunks(
                            session_id=st.session_state.session_id,
                            documents=st.session_state.chunks
                        )
                        st.success(
                            f"🎉 **ChromaDB Indexing Complete!** Added **{index_res['added']}** new chunk(s) "
                            f"({index_res['skipped']} duplicate(s) skipped). "
                            f"Total indexed chunks: **{index_res['total_in_collection']}**"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ **Embedding Error:** {str(exc)}")
