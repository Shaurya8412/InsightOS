"""
Streamlit Web Application: InsightOS User Interface.
"""

from __future__ import annotations

import logging
import streamlit as st

from src.core.config import settings
from src.frontend.api_client import APIClient, APIClientError

logger = logging.getLogger(__name__)

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="InsightOS — Grounded Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Premium Styling (Dark glassmorphism theme with Inter typography)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* General styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* App background */
    .stApp {
        background-color: #090D16;
        color: #E2E8F0;
    }
    
    /* Custom Headers */
    .app-title {
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Conversational Chat Bubbles */
    .chat-user {
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    
    .chat-assistant {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid #8B5CF6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    
    /* Citation visual design */
    .citation-container {
        margin-top: 0.8rem;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 6px;
    }
    
    .citation-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #38BDF8;
        margin-bottom: 0.4rem;
    }
    
    .citation-text {
        font-size: 0.85rem;
        color: #CBD5E1;
        font-style: italic;
        line-height: 1.4;
    }
    
    .citation-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    
    /* Form fields and buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, opacity 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Resolve API Client
api_url = settings.INSIGHTOS_API_URL
client = APIClient(api_url)

# Retrieve current document list from backend
try:
    documents = client.get_documents()
except Exception as exc:
    logger.error(f"Failed to load document library: {exc}")
    documents = []


# ---------------------------------------------------------------------------
# Sidebar Ingestion Layout
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔮 Ingestion Center")
    st.markdown("Upload files into the vector database for RAG retrieval.")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt", "md"],
        help="Supports PDF, Markdown, and flat text documents."
    )

    if uploaded_file is not None:
        # Determine if we already processed this file in the backend
        is_already_uploaded = any(
            doc["filename"] == uploaded_file.name for doc in documents
        )

        if not is_already_uploaded:
            with st.spinner("Parsing and indexing document chunks..."):
                try:
                    file_bytes = uploaded_file.read()
                    client.upload_document(uploaded_file.name, file_bytes)
                    st.success("🎉 Indexed successfully!")
                    st.rerun()
                except APIClientError as exc:
                    st.error(f"Ingestion failed: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # Display status list of uploaded documents
    if documents:
        st.markdown("---")
        st.markdown("#### 📄 Document Library")
        for doc in documents:
            status_color = "#10B981" if doc["status"] == "indexed" else "#EF4444" if doc["status"] == "failed" else "#F59E0B"
            size_kb = doc["file_size"] / 1024.0
            
            with st.container():
                st.markdown(
                    f"""
                    <div class='glass-card' style='padding: 0.8rem; margin-bottom: 0.5rem;'>
                        <strong style='font-size: 0.9rem; color: #E2E8F0;'>{doc['filename']}</strong><br/>
                        <span style='font-size: 0.75rem; color: #94A3B8;'>Size: {size_kb:.1f} KB | Chunks: {doc['chunk_count']}</span><br/>
                        <span style='font-size: 0.75rem; color: {status_color}; font-weight: 600;'>Status: {doc['status'].upper()}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Delete button with unique key matching document ID
                if st.button("Delete 🗑️", key=f"del_{doc['document_id']}"):
                    with st.spinner("Deleting document and vectors..."):
                        try:
                            client.delete_document(doc["document_id"])
                            st.success("Deleted successfully!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Deletion failed: {exc}")

    # RAG parameters
    st.markdown("---")
    st.markdown("#### ⚙️ Retrieval Options")
    top_k = st.slider("Top K chunks to retrieve", min_value=1, max_value=10, value=settings.RETRIEVAL_TOP_K)


# ---------------------------------------------------------------------------
# Main Chat Application Interface
# ---------------------------------------------------------------------------

st.markdown("<div class='app-title'>InsightOS RAG Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>A citation-aware question-answering assistant grounded in your uploaded documents.</div>", unsafe_allow_html=True)

# Main container for chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"<div class='chat-user'><strong>You:</strong><br/>{message['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            # Assistant answer
            st.markdown(
                f"<div class='chat-assistant'><strong>InsightOS:</strong><br/>{message['text']}</div>",
                unsafe_allow_html=True
            )
            
            # Citations display
            if message.get("citations"):
                st.markdown("<div style='margin-left: 1.5rem; margin-top: -0.4rem;'>", unsafe_allow_html=True)
                for idx, citation in enumerate(message["citations"], start=1):
                    # Format page info
                    page_info = f"Page {citation['page_number']}" if citation.get("page_number") is not None else ""
                    loc_info = citation.get("source_location") or ""
                    meta_details = " — ".join(filter(None, [citation["document_name"], page_info, loc_info]))
                    
                    with st.expander(f"[{idx}] {citation['document_name']}"):
                        st.markdown(
                            f"""
                            <div class='citation-container'>
                                <div class='citation-title'>
                                    <span class='citation-badge'>Source Details</span>
                                    {meta_details}
                                </div>
                                <div class='citation-text'>
                                    "... {citation['snippet']} ..."
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                st.markdown("</div>", unsafe_allow_html=True)

# User Chat Input
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input(
        "Ask a question about your documents...",
        placeholder="e.g. What is the architecture of RAG?"
    )
    submit_button = st.form_submit_method = st.form_submit_button("Send Query")

if submit_button and user_query.strip():
    # 1. Store user query
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    
    # 2. Call query endpoint
    with st.spinner("Synthesizing grounded response..."):
        try:
            response_model = client.query_rag(user_query, top_k=top_k)
            
            # Map Pydantic citation models to dictionaries for JSON session state stability
            citations_list = []
            for cit in response_model.citations:
                citations_list.append({
                    "chunk_id": str(cit.chunk_id),
                    "document_id": str(cit.document_id),
                    "document_name": cit.document_name,
                    "page_number": cit.page_number,
                    "source_location": cit.source_location,
                    "snippet": cit.snippet
                })

            st.session_state.chat_history.append({
                "role": "assistant",
                "text": response_model.answer,
                "citations": citations_list
            })
            
            # Trigger session rerun to render the new state
            st.rerun()
        except APIClientError as exc:
            st.error(f"Query execution failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected query error: {exc}")
