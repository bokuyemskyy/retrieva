from rag import RAG
import streamlit as st
from ui.state import init_session_state
from ui.sidebar import render_sidebar
from views.chat import render_chat
from views.files import render_files
from views.map import render_map

st.set_page_config(
    page_title="Retrieva",
    layout="wide",
    initial_sidebar_state="expanded",
)


init_session_state()
view = render_sidebar()

if view == "chat":
    render_chat()
elif view == "files":
    render_files()
elif view == "map":
    render_map()
