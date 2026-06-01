import streamlit as st


def render_files():
    ws = st.session_state.active_ws

    col_title, _ = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<p class="mono" style="font-size:0.75rem;color:var(--text-muted);letter-spacing:0.08em;margin-bottom:0;">FILES</p>'
            f'<h2 style="margin-top:0;margin-bottom:1.5rem;">{ws or "—"}</h2>',
            unsafe_allow_html=True,
        )

    if not ws:
        st.info("Select a workspace first.")
        return

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search",
            placeholder="Find relevant files…",
            label_visibility="collapsed",
            key="file_search_query",
        )
    with btn_col:
        search_clicked = st.button(
            "Search", use_container_width=True, key="file_search_btn"
        )

    st.divider()

    if search_clicked and query:
        with st.spinner("Retrieving…"):
            try:
                chunks = st.session_state.rag.retrieve_chunks(query, top_k=10)
                seen = set()
                results = []
                for chunk in chunks:
                    src = getattr(chunk, "source", None) or chunk.get(
                        "source", "unknown"
                    )
                    if src not in seen:
                        seen.add(src)
                        results.append({"source": src, "chunk": chunk})
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                results = []

        if results:
            st.markdown(
                f'<p class="section-label">Top results for "{query}"</p>',
                unsafe_allow_html=True,
            )
            for r in results:
                with st.expander(r["source"], expanded=False):
                    content = getattr(r["chunk"], "content", None) or r["chunk"].get(
                        "content", ""
                    )
                    st.markdown(
                        f'<span style="font-size:0.85rem;color:var(--text-muted);">{content[:400]}{"…" if len(content) > 400 else ""}</span>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No results found.")
    else:
        try:
            docs = st.session_state.rag.list_documents()
        except Exception:
            docs = []

        if not docs:
            st.markdown(
                '<p style="color:var(--text-muted);font-size:0.9rem;">No files indexed in this workspace.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p class="section-label">{len(docs)} document(s)</p>',
                unsafe_allow_html=True,
            )
            for doc in docs:
                name = getattr(doc, "name", None) or doc.get("name", str(doc))
                added = getattr(doc, "added_at", None) or doc.get("added_at", "")
                chunks_n = getattr(doc, "chunk_count", None) or doc.get(
                    "chunk_count", "—"
                )

                col_name, col_chunks, col_date, col_del = st.columns([5, 1, 2, 0.7])
                with col_name:
                    st.markdown(
                        f'<span class="mono" style="font-size:0.82rem;">{name}</span>',
                        unsafe_allow_html=True,
                    )
                with col_chunks:
                    st.markdown(
                        f'<span style="font-size:0.82rem;color:var(--text-muted);">{chunks_n} chunks</span>',
                        unsafe_allow_html=True,
                    )
                with col_date:
                    st.markdown(
                        f'<span style="font-size:0.82rem;color:var(--text-muted);">{str(added)[:10]}</span>',
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                        try:
                            st.session_state.rag.remove_document(name)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                st.markdown(
                    '<div style="border-top:1px solid var(--border);margin:0.3rem 0;"></div>',
                    unsafe_allow_html=True,
                )
