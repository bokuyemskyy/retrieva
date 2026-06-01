import streamlit as st
from retrieval.llm import LLMConfig
from ingestion.image_captioner import VLMConfig
from embedding.embedder import EmbedderConfig


@st.dialog("New workspace")
def create_workspace_dialog():
    name = st.text_input("Name", placeholder="my-workspace")

    st.markdown(
        '<p class="section-label" style="margin-top:0.75rem;">Embedding model</p>',
        unsafe_allow_html=True,
    )
    provider = st.selectbox("Provider", ["ollama", "openai"], key="ws_embed_prov")

    default_model = (
        "nomic-embed-text" if provider == "ollama" else "text-embedding-3-small"
    )
    model_name = st.text_input("Model name", value=default_model, key="ws_embed_model")

    default_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = st.text_input("Base URL", value=default_url, key="ws_embed_url")

    api_key = ""
    if provider == "openai":
        api_key = st.text_input("API key", type="password", key="ws_embed_key")

    if st.button("Create", type="primary", use_container_width=True):
        if not name:
            st.error("Name required.")
            return
        if name in st.session_state.workspaces:
            st.error("Workspace already exists.")
            return

        embed_cfg = EmbedderConfig(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or None,
        )
        try:
            st.session_state.rag.create_workspace(name=name, embedder_config=embed_cfg)
            st.session_state.rag.select_workspace(name)
        except Exception as e:
            st.error(f"RAG error: {e}")
            return

        st.session_state.workspaces.append(name)
        st.session_state.active_ws = name
        if name not in st.session_state.messages:
            st.session_state.messages[name] = []
        st.rerun()


@st.dialog("Add LLM")
def add_llm_dialog():
    provider = st.selectbox("Provider", ["ollama", "openai"], key="llm_add_prov")

    default_model = "llama3.2" if provider == "ollama" else "gpt-4o"
    model_name = st.text_input("Model name", value=default_model, key="llm_add_model")

    default_url = (
        "http://localhost:11434/v1"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = st.text_input("Base URL", value=default_url, key="llm_add_url")

    api_key = ""
    if provider == "openai":
        api_key = st.text_input("API key", type="password", key="llm_add_key")

    display_name = f"{model_name} ({provider})"

    if st.button("Add", type="primary", use_container_width=True):
        if not model_name:
            st.error("Model name required.")
            return

        cfg = {
            "name": display_name,
            "provider": provider,
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key or None,
        }
        st.session_state.llm_configs.append(cfg)
        st.session_state.active_llm = display_name

        try:
            llm_cfg = LLMConfig(
                provider=provider, model_name=model_name, base_url=base_url
            )
            st.session_state.rag.set_llm(llm_cfg)
        except Exception as e:
            st.warning(f"LLM not applied: {e}")

        st.rerun()


@st.dialog("Add VLM")
def add_vlm_dialog():
    provider = st.selectbox("Provider", ["ollama", "openai"], key="vlm_add_prov")

    default_model = "moondream" if provider == "ollama" else "gpt-4o"
    model_name = st.text_input("Model name", value=default_model, key="vlm_add_model")

    default_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = st.text_input("Base URL", value=default_url, key="vlm_add_url")

    api_key = ""
    if provider == "openai":
        api_key = st.text_input("API key", type="password", key="vlm_add_key")

    display_name = f"{model_name} ({provider})"

    if st.button("Add", type="primary", use_container_width=True):
        if not model_name:
            st.error("Model name required.")
            return

        cfg = {
            "name": display_name,
            "provider": provider,
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key or None,
        }
        st.session_state.vlm_configs.append(cfg)
        st.session_state.active_vlm = display_name

        try:
            vlm_cfg = VLMConfig(
                provider=provider, model_name=model_name, base_url=base_url
            )
            st.session_state.rag.set_vlm(vlm_cfg)
        except Exception as e:
            st.warning(f"VLM not applied: {e}")

        st.rerun()
