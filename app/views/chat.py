import streamlit as st
from ui.modals import add_llm_dialog, add_vlm_dialog


def _apply_llm(name: str):
    cfg = next((c for c in st.session_state.llm_configs if c["name"] == name), None)
    if not cfg:
        return
    try:
        from retrieval.llm import LLMConfig

        st.session_state.rag.set_llm(
            LLMConfig(
                provider=cfg["provider"],
                model_name=cfg["model_name"],
                base_url=cfg["base_url"],
            )
        )
    except Exception:
        pass


def _apply_vlm(name: str):
    cfg = next((c for c in st.session_state.vlm_configs if c["name"] == name), None)
    if not cfg:
        return
    try:
        from ingestion.image_captioner import VLMConfig

        st.session_state.rag.set_vlm(
            VLMConfig(
                provider=cfg["provider"],
                model_name=cfg["model_name"],
                base_url=cfg["base_url"],
            )
        )
    except Exception:
        pass


def render_chat():
    ws = st.session_state.active_ws

    col_title, col_spacer = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<p class="mono" style="font-size:0.75rem;color:var(--text-muted);letter-spacing:0.08em;margin-bottom:0;">WORKSPACE</p>'
            f'<h2 style="margin-top:0;margin-bottom:1.5rem;">{ws or "—"}</h2>',
            unsafe_allow_html=True,
        )

    if not ws:
        st.info("Create or select a workspace to start chatting.")
        return

    if ws not in st.session_state.messages:
        st.session_state.messages[ws] = []

    llm_names = [c["name"] for c in st.session_state.llm_configs]
    vlm_names = [c["name"] for c in st.session_state.vlm_configs]

    ctrl1, ctrl2, ctrl3, ctrl4, _ = st.columns([2, 0.5, 2, 0.5, 3])

    with ctrl1:
        if llm_names:
            cur_idx = (
                llm_names.index(st.session_state.active_llm)
                if st.session_state.active_llm in llm_names
                else 0
            )
            chosen_llm = st.selectbox(
                "LLM",
                llm_names,
                index=cur_idx,
                label_visibility="collapsed",
                key="llm_select",
            )
            if chosen_llm != st.session_state.active_llm:
                st.session_state.active_llm = chosen_llm
                _apply_llm(chosen_llm)
                st.rerun()
        else:
            st.caption("No LLM configured")

    with ctrl2:
        if st.button("+", key="add_llm_btn", help="Add LLM"):
            add_llm_dialog()

    with ctrl3:
        if vlm_names:
            cur_idx = (
                vlm_names.index(st.session_state.active_vlm)
                if st.session_state.active_vlm in vlm_names
                else 0
            )
            chosen_vlm = st.selectbox(
                "VLM",
                vlm_names,
                index=cur_idx,
                label_visibility="collapsed",
                key="vlm_select",
            )
            if chosen_vlm != st.session_state.active_vlm:
                st.session_state.active_vlm = chosen_vlm
                _apply_vlm(chosen_vlm)
                st.rerun()
        else:
            st.caption("No VLM configured")

    with ctrl4:
        if st.button("+", key="add_vlm_btn", help="Add VLM"):
            add_vlm_dialog()

    with st.expander("Add files to workspace", expanded=False):
        uploaded = st.file_uploader(
            "Drop files here (PDF, TXT, MD…)",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Index files", type="primary"):
            with st.spinner("Ingesting…"):
                for f in uploaded:
                    try:
                        import tempfile, os

                        suffix = os.path.splitext(f.name)[1]
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        st.session_state.rag.add_document(tmp_path)
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
            st.success(f"Indexed {len(uploaded)} file(s).")

    st.divider()

    messages = st.session_state.messages[ws]
    chat_box = st.container(height=480)
    with chat_box:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask the knowledge base...")
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                llm_label = st.session_state.active_llm or "LLM"
                with st.spinner(f"Querying {llm_label}…"):
                    try:
                        response = st.session_state.rag.query(prompt)
                    except Exception as e:
                        response = f"*Error: {e}*"
                st.markdown(response)
        messages.append({"role": "assistant", "content": response})
