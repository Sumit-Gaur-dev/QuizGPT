"""
Quiz RAG - Application Entry Point
Phase 1: Streamlit UI Shell & Modular Layout
"""

import streamlit as st
from config.session import init_session_state, reset_session_data
from frontend.dashboard import render_dashboard
from frontend.upload import render_upload
from frontend.generate_quiz import render_generate_quiz
from frontend.quiz_attempt import render_quiz_attempt
from frontend.quiz_results import render_quiz_history

# 1. Page Configuration
st.set_page_config(
    page_title="Quiz RAG - AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
init_session_state()

# Navigation pages list
PAGES = [
    "📊 Dashboard",
    "📁 Upload Notes",
    "⚙️ Generate Quiz",
    "✍️ Attempt Quiz",
    "📜 Quiz History"
]

# Ensure current_page is valid
if st.session_state.current_page not in PAGES:
    st.session_state.current_page = PAGES[0]

# Callback to handle sidebar clicks
def on_nav_change():
    st.session_state.current_page = st.session_state.nav_selection

# 3. Sidebar UI
with st.sidebar:
    st.title("📚 Quiz RAG")
    st.caption("AI-Powered Study Quiz Assistant • Phase 1")

    st.divider()

    # Navigation menu
    current_index = PAGES.index(st.session_state.current_page)
    st.radio(
        label="Navigation Menu",
        options=PAGES,
        index=current_index,
        key="nav_selection",
        on_change=on_nav_change
    )

    st.divider()

    # Session Status Indicators
    st.subheader("📌 Session Info")
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    num_files = len(st.session_state.uploaded_files)
    st.caption(f"Notes Uploaded: **{num_files}**")
    has_quiz = "Yes" if st.session_state.current_quiz else "None"
    st.caption(f"Active Quiz: **{has_quiz}**")

    st.divider()

    # Data Lifecycle: Delete My Data
    st.subheader("🛡️ Data & Privacy")
    st.caption("All session data is temporary and isolated to your browser session.")
    if st.button("🗑️ Delete My Data", use_container_width=True, help="Clear all uploaded documents and quiz history"):
        reset_session_data()
        st.toast("Session data cleared successfully!", icon="🧹")
        st.rerun()

# 4. Main Body Rendering
if st.session_state.current_page == "📊 Dashboard":
    render_dashboard()
elif st.session_state.current_page == "📁 Upload Notes":
    render_upload()
elif st.session_state.current_page == "⚙️ Generate Quiz":
    render_generate_quiz()
elif st.session_state.current_page == "✍️ Attempt Quiz":
    render_quiz_attempt()
elif st.session_state.current_page == "📜 Quiz History":
    render_quiz_history()
