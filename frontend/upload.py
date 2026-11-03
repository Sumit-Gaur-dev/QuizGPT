"""
Upload section view for Quiz RAG.
Handles PDF file uploads and displays placeholder ingestion status (Phase 1 mock).
"""

import streamlit as st


def render_upload() -> None:
    """Render the PDF upload and file management section."""
    st.header("📁 Upload Study Notes")
    st.markdown(
        "Upload one or more PDF notes. In Phase 1, files are registered in the UI state with "
        "a mock ingestion status. Full PyMuPDF text parsing will be connected in Phase 2."
    )

    st.divider()

    # Upload widget
    uploaded_files = st.file_uploader(
        label="Select PDF files from your computer",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based PDF lecture notes or reading materials (V1 supports text PDFs, not OCR)."
    )

    if uploaded_files:
        existing_names = {f["name"] for f in st.session_state.uploaded_files}
        new_count = 0

        for file in uploaded_files:
            if file.name not in existing_names:
                st.session_state.uploaded_files.append({
                    "id": f"upload_{file.name}",
                    "name": file.name,
                    "size_kb": round(file.size / 1024, 1),
                    "pages": "(Pending Phase 2 extraction)",
                    "status": "Ready for Indexing (Mock)"
                })
                new_count += 1

        if new_count > 0:
            st.success(f"Successfully added {new_count} new file(s) to the session state!")

    st.markdown("### 📋 Uploaded Files & Ingestion Status")

    if not st.session_state.uploaded_files:
        st.warning("No PDF notes currently uploaded. Upload a PDF above to begin.")
    else:
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            with st.container(border=True):
                col_name, col_size, col_status, col_action = st.columns([3, 1.5, 2.5, 1])

                with col_name:
                    st.markdown(f"📄 **{file_info['name']}**")

                with col_size:
                    st.caption(f"Size: `{file_info['size_kb']} KB`")

                with col_status:
                    st.info(f"Status: **{file_info['status']}**")

                with col_action:
                    if st.button("Remove", key=f"remove_file_{idx}", use_container_width=True):
                        st.session_state.uploaded_files.pop(idx)
                        st.rerun()

    st.markdown("---")

    # Informational notice regarding Phase 1 vs Phase 2
    with st.expander("ℹ️ Phase 1 Note: Ingestion Pipeline Status", expanded=True):
        st.markdown(
            """
            - **Current Phase (Phase 1):** Mock upload state. File metadata and placeholders are stored in `st.session_state`.
            - **Upcoming Phase (Phase 2):** PyMuPDF (`fitz`) will read each PDF, detect empty/scanned pages, and extract clean text chunks with page numbers.
            """
        )

