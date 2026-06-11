import streamlit as st
from core.embedding.embedder import EmbedderConfig

NAV_ITEMS = [
    ("chat", "Chat"),
    ("documents", "Documents"),
    ("models", "Models"),
    ("projection", "Projection"),
]


@st.dialog("New workspace")
def _create_workspace_dialog():
    name = st.text_input("Name", placeholder="my-workspace")
    provider = st.selectbox("Embedding provider", ["ollama", "openai"])
    default_model = (
        "nomic-embed-text" if provider == "ollama" else "text-embedding-3-small"
    )
    model_name = st.text_input("Model name", value=default_model)
    default_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = st.text_input("Base URL", value=default_url)
    api_key = st.text_input("API key", type="password") if provider == "openai" else ""

    if st.button("Create", type="primary", use_container_width=True):
        if not name:
            st.error("Name required.")
            return
        try:
            st.session_state.rag.create_workspace(
                name=name,
                embedder_config=EmbedderConfig(
                    provider=provider,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key or None,
                ),
            )
            st.session_state.active_ws = name
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


@st.dialog("Confirm deletion")
def _delete_workspace_dialog(name: str):
    st.write(f"Delete workspace **{name}**?")
    col1, col2 = st.columns(2)
    if col1.button("Delete", type="primary", use_container_width=True):
        try:
            st.session_state.rag.delete_workspace(name)
            st.session_state.active_ws = None
        except Exception as e:
            st.error(str(e))
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("Confirm deletion")
def _delete_all_dialog():
    st.write("Delete **all** workspaces?")
    col1, col2 = st.columns(2)
    if col1.button("Delete all", type="primary", use_container_width=True):
        try:
            st.session_state.rag.delete_all_workspaces()
            st.session_state.active_ws = None
        except Exception as e:
            st.error(str(e))
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.rerun()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("# Retrieva")
        st.divider()

        st.markdown("### Navigation")
        for key, label in NAV_ITEMS:
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.active_view = key
                st.rerun()

        st.divider()
        st.markdown("### Workspace")

        try:
            workspaces = st.session_state.rag.get_workspaces()
            ws_names = [ws.name for ws in workspaces]
        except Exception:
            ws_names = []

        if ws_names:
            idx = (
                ws_names.index(st.session_state.active_ws)
                if st.session_state.active_ws in ws_names
                else 0
            )
            selected = st.selectbox(
                "Select", ws_names, index=idx, label_visibility="collapsed"
            )
            if selected != st.session_state.active_ws:
                st.session_state.active_ws = selected
                st.rerun()
        else:
            st.caption("No workspaces yet.")

        if st.button("New workspace", use_container_width=True):
            _create_workspace_dialog()

        if ws_names:
            col1, col2 = st.columns(2)
            if col1.button(
                "Delete",
                use_container_width=True,
                disabled=not st.session_state.active_ws,
            ):
                _delete_workspace_dialog(st.session_state.active_ws)
            if col2.button("Delete all", use_container_width=True):
                _delete_all_dialog()

    return st.session_state.active_view
