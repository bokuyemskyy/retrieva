import streamlit as st
from ui.config_helpers import get_active_llm_config


def render_chat():
    st.title("Chat")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    llm_cfg = get_active_llm_config()
    if not llm_cfg:
        st.warning("No LLM selected. Please select or add one on the Models page.")
        return

    if ws not in st.session_state.messages:
        st.session_state.messages[ws] = []
    messages = st.session_state.messages[ws]

    chat_container = st.container(height=480)
    with chat_container:
        for msg in messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask about your documents...")
    if query:
        llm_cfg = get_active_llm_config()
        if not llm_cfg:
            st.error("No LLM configured. Go to Models page to add one.")
            return

        messages.append({"role": "user", "content": query})
        with chat_container:
            st.chat_message("user").markdown(query)
            with st.chat_message("assistant"):
                try:
                    stream = st.session_state.rag.query_stream(
                        workspace_name=ws,
                        query=query,
                        llm_config=llm_cfg,
                    )
                    response = st.write_stream(stream)
                except Exception as e:
                    response = f"*Error: {e}*"
                    st.markdown(response)

        messages.append({"role": "assistant", "content": response})
