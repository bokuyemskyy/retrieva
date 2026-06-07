import subprocess
import sys

import streamlit as st
from ui.config_helpers import get_active_vlm_config


def _open_file(path: str):
    try:
        if sys.platform == "win32":
            subprocess.Popen(["start", "", path], shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        st.error(f"Could not open file: {e}")


def render_documents():
    st.title("Documents")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    col_upload, col_btn = st.columns([5, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Add documents",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "wav", "mp3"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )
    with col_btn:
        st.markdown("<div style='padding-top:1.9rem'>", unsafe_allow_html=True)
        index_clicked = st.button(
            "Index files",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if index_clicked and uploaded_files:
        vlm_cfg = get_active_vlm_config()
        with st.spinner(f"Ingesting {len(uploaded_files)} file(s)…"):
            try:
                docs = [(f.name, f.getvalue()) for f in uploaded_files]
                st.session_state.rag.add_documents(
                    workspace_name=ws,
                    documents=docs,
                    vlm_config=vlm_cfg,
                )
                st.session_state.uploader_key += 1
                st.success(f"Indexed {len(uploaded_files)} file(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.divider()

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder="Find relevant documents…",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("Search", use_container_width=True)

    st.divider()

    if search_clicked and query:
        with st.spinner("Searching…"):
            try:
                results = st.session_state.rag.search_files(
                    workspace_name=ws, query=query, top_k=10
                )
            except Exception as e:
                st.error(f"Search error: {e}")
                results = []
        if not results:
            st.info("No matching documents found.")
        else:
            st.caption(f'{len(results)} result(s) for "{query}"')
            _render_doc_grid(results, scored=True)
    else:
        try:
            docs = st.session_state.rag.get_documents(workspace_name=ws)
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            docs = []
        if not docs:
            st.info("No documents in this workspace yet.")
        else:
            st.caption(f"{len(docs)} document(s)")
            _render_doc_grid(docs, scored=False)


def _render_doc_grid(items, scored: bool):
    cols = st.columns(3)
    for i, item in enumerate(items):
        doc = item.document if scored else item
        score = item.score if scored else None

        with cols[i % 3]:
            score_line = f"score: {score:.3f}" if score is not None else "&nbsp;"
            st.markdown(
                f"""<div style="
                    border: 1px solid rgba(128,128,128,0.3);
                    border-radius: 6px;
                    padding: 0.6rem 0.8rem;
                    height: 72px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    margin-bottom: 0.25rem;
                    overflow: hidden;
                ">
                    <div style="font-size:0.85rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">📄 {doc.filename}</div>
                    <div style="font-size:0.75rem;opacity:0.55;">{score_line}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "Open", key=f"doc_{doc.document_id}", use_container_width=True
            ):
                _open_file(doc.source_path)
