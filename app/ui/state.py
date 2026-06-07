import json
from pathlib import Path

import streamlit as st

from core.rag import RAG

STORAGE_DIR = "./storage"
POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
CONFIGS_FILE = Path("./retrieva_configs.json")


def _load_persisted_configs() -> dict:
    if CONFIGS_FILE.exists():
        try:
            return json.loads(CONFIGS_FILE.read_text())
        except Exception:
            pass
    return {"llm_configs": [], "vlm_configs": []}


def save_configs():
    data = {
        "llm_configs": st.session_state.llm_configs,
        "vlm_configs": st.session_state.vlm_configs,
        "active_llm": st.session_state.active_llm,
        "active_vlm": st.session_state.active_vlm,
    }
    CONFIGS_FILE.write_text(json.dumps(data, indent=2))


@st.cache_resource
def get_rag() -> RAG:
    return RAG(storage_dir=STORAGE_DIR, postgres_url=POSTGRES_URL)


def init_session_state():
    if "rag" not in st.session_state:
        st.session_state.rag = get_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if "active_ws" not in st.session_state:
        st.session_state.active_ws = None

    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"

    if "configs_loaded" not in st.session_state:
        persisted = _load_persisted_configs()
        st.session_state.llm_configs = persisted.get("llm_configs", [])
        st.session_state.vlm_configs = persisted.get("vlm_configs", [])
        st.session_state.active_llm = persisted.get("active_llm")
        st.session_state.active_vlm = persisted.get("active_vlm")
        st.session_state.configs_loaded = True
