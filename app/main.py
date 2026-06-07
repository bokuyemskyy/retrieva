import streamlit as st
from app.ui.state import init_session_state
from app.ui.sidebar import render_sidebar
from app.views.chat import render_chat
from app.views.documents import render_documents
from app.views.models import render_models

st.set_page_config(page_title="Retrieva")

init_session_state()
view = render_sidebar()

if view == "chat":
    render_chat()
elif view == "documents":
    render_documents()
elif view == "models":
    render_models()
