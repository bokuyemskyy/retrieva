import streamlit as st
from ui.state import save_configs


def _model_form(
    prefix: str,
    default_model_ollama: str,
    default_model_openai: str,
    config_key: str,
    active_key: str,
):
    provider = st.selectbox("Provider", ["ollama", "openai"], key=f"{prefix}_prov")
    default_model = (
        default_model_ollama if provider == "ollama" else default_model_openai
    )
    model_name = st.text_input("Model name", value=default_model, key=f"{prefix}_model")
    default_url = (
        "http://localhost:11434"
        if provider == "ollama"
        else "https://api.openai.com/v1"
    )
    base_url = st.text_input("Base URL", value=default_url, key=f"{prefix}_url")
    api_key = (
        st.text_input("API key", type="password", key=f"{prefix}_key")
        if provider == "openai"
        else ""
    )

    if st.button("Add", type="primary", use_container_width=True, key=f"{prefix}_add"):
        if not model_name:
            st.error("Model name required.")
            return

        cfg = {
            "name": f"{model_name} ({provider})",
            "provider": provider,
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key or None,
        }
        st.session_state[config_key].append(cfg)
        st.session_state[active_key] = cfg["name"]
        save_configs()
        st.rerun()


@st.dialog("Add LLM")
def _add_llm_dialog():
    _model_form("llm_dlg", "llama3.2", "gpt-4o-mini", "llm_configs", "active_llm")


@st.dialog("Add VLM")
def _add_vlm_dialog():
    _model_form("vlm_dlg", "moondream", "gpt-4o", "vlm_configs", "active_vlm")


@st.dialog("Confirm deletion")
def _delete_model_dialog(name: str, config_key: str, active_key: str):
    st.write(f"Delete model **{name}**?")
    col1, col2 = st.columns(2)
    if col1.button(
        "Delete", type="primary", use_container_width=True, key=f"del_confirm_{name}"
    ):
        configs = st.session_state[config_key]
        st.session_state[config_key] = [c for c in configs if c["name"] != name]

        if st.session_state[active_key] == name:
            st.session_state[active_key] = (
                st.session_state[config_key][0]["name"]
                if st.session_state[config_key]
                else None
            )
        save_configs()
        st.rerun()
    if col2.button("Cancel", use_container_width=True, key=f"del_cancel_{name}"):
        st.rerun()


@st.dialog("Confirm deletion")
def _delete_all_models_dialog(config_key: str, active_key: str):
    st.write("Delete **all** models in this category?")
    col1, col2 = st.columns(2)
    if col1.button(
        "Delete all",
        type="primary",
        use_container_width=True,
        key=f"del_all_confirm_{config_key}",
    ):
        st.session_state[config_key] = []
        st.session_state[active_key] = None
        save_configs()
        st.rerun()
    if col2.button(
        "Cancel", use_container_width=True, key=f"del_all_cancel_{config_key}"
    ):
        st.rerun()


def _render_model_selection(config_key: str, active_key: str, label: str) -> bool:
    configs = st.session_state.get(config_key, [])
    if not configs:
        st.caption(f"No {label} configured.")
        return False

    names = [c["name"] for c in configs]
    active_val = st.session_state.get(active_key)
    idx = names.index(active_val) if active_val in names else 0

    selected = st.selectbox(
        f"Select {label}",
        names,
        index=idx,
        label_visibility="collapsed",
        key=f"sel_{label}",
    )

    if selected != active_val:
        st.session_state[active_key] = selected
        save_configs()
        st.rerun()

    return True


def render_models():
    st.title("Models")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    col_llm, col_vlm = st.columns(2)

    with col_llm:
        st.subheader("LLM")
        has_llm = _render_model_selection("llm_configs", "active_llm", "LLM")

        if st.button("Add LLM", use_container_width=True):
            _add_llm_dialog()

        col1, col2 = st.columns(2)
        if col1.button(
            "Delete", use_container_width=True, disabled=not has_llm, key="del_btn_llm"
        ):
            _delete_model_dialog(
                st.session_state.active_llm, "llm_configs", "active_llm"
            )
        if col2.button(
            "Delete all",
            use_container_width=True,
            disabled=not has_llm,
            key="del_all_btn_llm",
        ):
            _delete_all_models_dialog("llm_configs", "active_llm")

    with col_vlm:
        st.subheader("VLM")
        has_vlm = _render_model_selection("vlm_configs", "active_vlm", "VLM")

        if st.button("Add VLM", use_container_width=True):
            _add_vlm_dialog()

        col1, col2 = st.columns(2)
        if col1.button(
            "Delete", use_container_width=True, disabled=not has_vlm, key="del_btn_vlm"
        ):
            _delete_model_dialog(
                st.session_state.active_vlm, "vlm_configs", "active_vlm"
            )
        if col2.button(
            "Delete all",
            use_container_width=True,
            disabled=not has_vlm,
            key="del_all_btn_vlm",
        ):
            _delete_all_models_dialog("vlm_configs", "active_vlm")
