from openai import OpenAI
from core.rag import RAG
from core.embedding.embedder import EmbedderConfig, OllamaEmbedder, EmbedderFactory
from core.ingestion.chunker import Chunker
from core.retrieval.llm import LLMConfig
from core.ingestion.image_captioner import VLMConfig
from core.ingestion.image_captioner import ImageCaptioner, OllamaVLM, VLMConfig, VLMFactory
import streamlit as st

def get_rag() -> RAG:
    if "rag" not in st.session_state:
        st.session_state.rag = RAG(
            storage_dir="./storage",
            postgres_url="postgresql://postgres:postgres@localhost:5432/postgres",
        )

        st.session_state.rag.set_llm(LLMConfig(provider="ollama", model_name="qwen2.5:3b"))
        st.session_state.rag.set_vlm(VLMConfig(provider="ollama", model_name="moondream:v2"))
    return st.session_state.rag

@st.dialog("Create new workspace")
def add_new_workspace():
    name = st.text_input(label="Workspace name", value="default")
    provider = st.selectbox(label="Provider", options=["ollama", "openai"])
    model_name = st.text_input(label="Model name", value="nomic-embed-text")
    base_url = st.text_input(label="Base URL", value="http://localhost:11434")
    api_key = st.text_input(label="API key")


    chunk_size = st.number_input(label="Chunk size", min_value=1, value=2048)
    chunk_overlap = st.number_input(label="Chunk overlap", min_value=1, value=256)

    if st.button("Create"):
        get_rag().create_workspace(name, EmbedderConfig(
            provider,
            model_name,
            chunk_size,
            chunk_overlap,
            api_key,
            base_url,
        ))
        st.rerun()


@st.dialog("Confirm deletion")
def delete_workspace(name: str):
    st.write(f"Confirm deletion of {name} workspace?")
    col1, col2 = st.columns(2)
    if col1.button("Yes", width="stretch"):
        get_rag().delete_workspace(name)
        st.rerun()
    if col2.button("No", width="stretch"):
        st.rerun()

@st.dialog("Confirm deletion")
def delete_all_workspaces():
    st.write("Confirm deletion of all workspaces?")
    col1, col2 = st.columns(2)
    if col1.button("Yes", width="stretch"):
        get_rag().delete_all_workspaces()
        st.rerun()
    if col2.button("No", width="stretch"):
        st.rerun()

def main():
    rag = get_rag()
    st.set_page_config(page_title="Retrieva", page_icon="🤖")
    st.title("Retrieva")

    workspaces = rag.get_workspaces()
    workspace = st.selectbox("Workspace", [workspace.name for workspace in workspaces])
    col1, col2, col3 = st.columns(3)
    if col1.button("Create new workspace", width="stretch"):
        add_new_workspace()
    if col2.button("Delete workspace", width="stretch", disabled=workspace is None):
        delete_workspace(workspace)
    if col3.button("Delete all workspaces", width="stretch", disabled=len(workspaces) == 0):
        delete_all_workspaces()

    if workspace:
        rag.select_workspace(workspace)

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        uploaded_files = st.file_uploader("Upload documents", type=[
            "pdf",
            "txt",
            "md",
            "png",
            "jpg",
            "jpeg",
            "wav",
            "mp3",
        ], accept_multiple_files=True, key=st.session_state.uploader_key)

        if uploaded_files:
            for file in uploaded_files:
                st.write(f"Adding {file.name}")
                rag.add_document((file.name, file.getvalue()))

            st.session_state.uploader_key += 1
            st.rerun()

        query = st.chat_input("Ask about something")
        if query:
            st.chat_message("user").write(query)
            result = rag.query(query)
            st.chat_message("ai").write(result)


if __name__ == "__main__":
    main()