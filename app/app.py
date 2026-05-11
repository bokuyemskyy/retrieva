import streamlit as st

from components.chat import render_chat
from components.uploader import render_sidebar

st.set_page_config(page_title="AGH-CUDA RAG Chatbot", page_icon="🤖", layout="wide")


def main():
    st.title("Knowledge Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        render_sidebar()

    render_chat()


if __name__ == "__main__":
    main()
