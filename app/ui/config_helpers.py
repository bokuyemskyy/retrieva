import streamlit as st
from core.retrieval.llm import LLMConfig
from core.ingestion.image_captioner import VLMConfig


def get_active_llm_config() -> LLMConfig | None:
    name = st.session_state.active_llm
    cfg = next((c for c in st.session_state.llm_configs if c["name"] == name), None)
    if not cfg:
        cfg = st.session_state.llm_configs[0] if st.session_state.llm_configs else None
    if not cfg:
        return None
    return LLMConfig(
        provider=cfg["provider"],
        model_name=cfg["model_name"],
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
    )


def get_active_vlm_config() -> VLMConfig | None:
    name = st.session_state.active_vlm
    cfg = next((c for c in st.session_state.vlm_configs if c["name"] == name), None)
    if not cfg:
        cfg = st.session_state.vlm_configs[0] if st.session_state.vlm_configs else None
    if not cfg:
        return None
    return VLMConfig(
        provider=cfg["provider"],
        model_name=cfg["model_name"],
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
    )
