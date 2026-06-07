import streamlit as st

from app.ui.config_helpers import get_active_vlm_config


def render_documents():
    st.title("Documents")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    uploaded_files = st.file_uploader(
        "Add documents to workspace",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg", "wav", "mp3"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    if uploaded_files:
        if st.button("Index files", type="primary"):
            vlm_cfg = get_active_vlm_config()
            with st.spinner(f"Ingesting {len(uploaded_files)} file(s)…"):
                try:
                    docs = [(f.name, f.getvalue()) for f in uploaded_files]
                    st.session_state.rag.add_documents(
                        workspace_name=ws,
                        documents=docs,
                        vlm_config=vlm_cfg,
                    )
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
            st.info(
                "No documents in this workspace yet. Upload some from the Chat page."
            )
        else:
            st.caption(f"{len(docs)} document(s)")
            _render_doc_grid(docs, scored=False)


def _render_doc_grid(items, scored: bool):
    cols = st.columns(3)
    for i, item in enumerate(items):
        doc = item.document if scored else item
        score = item.score if scored else None

        with cols[i % 3]:
            label = f"📄 {doc.filename}"
            if score is not None:
                label += f"  \n`score: {score:.3f}`"
            if st.button(label, key=f"doc_{doc.document_id}", use_container_width=True):
                # Open the source path in browser via a link
                st.markdown(
                    f'<a href="file://{doc.source_path}" target="_blank">Open file</a>',
                    unsafe_allow_html=True,
                )
