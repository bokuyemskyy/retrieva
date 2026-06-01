import streamlit as st
from core.rag import RAG

ST_STORAGE_DIR = "./rag_storage"
POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


def init_session_state():
    if "rag" not in st.session_state:
        st.session_state.rag = RAG(
            storage_dir=ST_STORAGE_DIR,
            postgres_url=POSTGRES_URL,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if "workspaces" not in st.session_state:
        st.session_state.workspaces = []

    if "active_ws" not in st.session_state:
        st.session_state.active_ws = None

    if "llm_configs" not in st.session_state:
        st.session_state.llm_configs = []

    if "active_llm" not in st.session_state:
        st.session_state.active_llm = None

    if "vlm_configs" not in st.session_state:
        st.session_state.vlm_configs = []

    if "active_vlm" not in st.session_state:
        st.session_state.active_vlm = None

    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"
