"""
Semantic Search and Document Retrieval View for Quiz RAG.
Allows users to enter natural-language study queries and inspect
the Top-K most relevant document chunks retrieved from ChromaDB.
"""

import streamlit as st
from config.settings import RETRIEVAL_TOP_K
from rag.embeddings.factory import get_active_provider_info
from rag.vector_store import get_collection_stats
from rag.retrieval import retrieve_chunks_with_scores


def render_search() -> None:
    """Render the semantic search and retrieval user interface."""
    st.header("🔍 Semantic Search & Document Retrieval")
    st.markdown(
        "Search your indexed study notes using natural language. Query embeddings are generated "
        "using the active embedding provider and matched against stored vectors in ChromaDB."
    )

    st.divider()

    # Active Provider & Collection Status
    provider_info = get_active_provider_info()
    stats = get_collection_stats(st.session_state.session_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Embedding Provider", value=provider_info["display_name"])
    with col2:
        st.metric(label="Embedding Model", value=provider_info["model"].split("/")[-1])
    with col3:
        st.metric(label="Indexed Chunks in ChromaDB", value=stats["total_chunks"])

    st.divider()

    # Empty Database Guardrail
    if stats["total_chunks"] == 0:
        st.warning(
            "⚠️ **Your ChromaDB vector database is currently empty.**\n\n"
            "To perform semantic search, please navigate to **📁 Upload Notes**, "
            "upload or load a note, and click **⚡ Generate Embeddings & Store in ChromaDB**."
        )
        if st.button("Go to Upload & Indexing ➔", type="primary"):
            st.session_state.current_page = "📁 Upload Notes"
            st.rerun()
        return

    # Sample query quick-fill suggestions
    st.caption("💡 Quick test queries (click to copy into search input):")
    sample_queries = [
        "What is Retrieval-Augmented Generation and its key benefits?",
        "How does text chunking and overlap preserve context?",
        "Why is RAG preferred over fine-tuning for factual recall?",
        "Deterministic evaluation vs LLM evaluation"
    ]
    
    sample_cols = st.columns(len(sample_queries))
    for idx, q_text in enumerate(sample_queries):
        with sample_cols[idx]:
            if st.button(f"Query #{idx + 1}", key=f"btn_sq_{idx}", use_container_width=True, help=q_text):
                st.session_state["search_query_input"] = q_text

    # Search Form
    with st.form("semantic_search_form"):
        query = st.text_input(
            label="Enter your study question or topic:",
            value=st.session_state.get("search_query_input", ""),
            placeholder="e.g., How does chunk overlap prevent loss of context across split boundaries?",
            help="Type natural language questions or concept keywords."
        )

        col_top_k, col_submit = st.columns([1, 2])
        with col_top_k:
            top_k = st.slider(
                label="Top-K Results",
                min_value=1,
                max_value=10,
                value=RETRIEVAL_TOP_K,
                help="Maximum number of relevant chunks to retrieve from the vector database."
            )

        with col_submit:
            st.write("") # vertical spacing
            submit_search = st.form_submit_button(
                "🔍 Search Vector Database",
                type="primary",
                use_container_width=True
            )

    # Perform Search upon Form Submission
    if submit_search:
        if not query or not query.strip():
            st.warning("Please enter a query before searching.")
            return

        with st.spinner(f"Embedding query and finding Top-{top_k} nearest chunks in ChromaDB..."):
            try:
                results = retrieve_chunks_with_scores(
                    session_id=st.session_state.session_id,
                    query=query.strip(),
                    top_k=top_k
                )
                st.session_state["last_search_query"] = query.strip()
                st.session_state["last_search_results"] = results
            except Exception as exc:
                st.error(f"❌ Retrieval error: {str(exc)}")
                return

    # Display Search Results
    last_query = st.session_state.get("last_search_query")
    results = st.session_state.get("last_search_results")

    if last_query and results is not None:
        st.markdown(f"### 📋 Retrieval Results for: *\"{last_query}\"*")

        if not results:
            st.info("No matching chunks found in the vector database.")
            return

        st.caption(f"Retrieved **{len(results)}** chunk(s) from ChromaDB collection `{stats['collection_name']}`:")

        # Distance interpretation notice
        st.info(
            "ℹ️ **Understanding ChromaDB Distance:** "
            "ChromaDB calculates the vector distance (L2 / squared Euclidean or Cosine distance) between the query vector "
            "and each chunk vector. **A lower distance indicates closer geometric proximity and higher semantic similarity.** "
            "(Distance is not a percentage)."
        )

        for rank, res in enumerate(results, start=1):
            with st.container(border=True):
                col_rank, col_meta, col_dist = st.columns([1, 4, 2])

                with col_rank:
                    st.markdown(f"### #{rank}")

                with col_meta:
                    st.markdown(f"📄 **{res['source']}**")
                    st.caption(f"Original Page: **{res['page']}** | Chunk ID: `{res['chunk_id']}` | Length: `{res['char_count']} chars`")

                with col_dist:
                    st.metric(
                        label="ChromaDB Distance",
                        value=f"{res['distance']:.4f}",
                        help="Lower distance indicates closer vector proximity to the query."
                    )

                # Chunk Content Preview
                st.markdown("**Retrieved Chunk Text:**")
                st.text_area(
                    label=f"Text of Result #{rank}",
                    value=res["text"],
                    height=100,
                    disabled=True,
                    key=f"search_res_text_{rank}",
                    label_visibility="collapsed"
                )

                with st.expander("Full Chunk Metadata JSON", expanded=False):
                    st.json(res["metadata"])

