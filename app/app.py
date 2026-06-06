from openai import OpenAI
from core.rag import RAG
from core.embedding.embedder import EmbedderConfig, OllamaEmbedder
from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner, OllamaVLM
import streamlit as st

def main():
    st.set_page_config(page_title="Retrieva", page_icon="🤖")

    llm = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )

    chunker = Chunker(
        chunk_size=512,
        chunk_overlap=128,
    )

    image_captioner = ImageCaptioner(
        vlm=OllamaVLM(model_name="moondream:v2"), 
        use_ocr=True
    )

    rag = RAG(
        storage_dir="./storage",
        llm_client=llm,
        llm_model="qwen2.5:3b",  
        chunker=chunker,
        image_captioner=image_captioner,
        postgres_url="postgresql://postgres:postgres@localhost:5432/postgres",
    )

    embedder_config = OllamaEmbedder(
        config=EmbedderConfig(
            provider="ollama",
            model_name="qllama/bge-large-en-v1.5:q8_0",
        )
    )

    st.title("Retrieva")

    # rag.create_workspace("test3", embedder_config)
    workspaces = rag.get_workspaces()
    workspace = st.selectbox("Workspace", [workspace.name for workspace in workspaces])
    rag.select_workspace(workspace)

    uploaded_files = st.file_uploader("Upload documents", type=[
        "pdf",
        "txt",
        "md",
        "png",
        "jpg",
        "jpeg",
        "wav",
        "mp3",
    ], accept_multiple_files=True)
    # for file in uploaded_files:
        # st.write(file.name)
        # rag.add_document()

    query = st.chat_input("Ask about something")
    if query:
        st.chat_message("user").write(query)
        result = rag.query(query)
        st.chat_message("ai").write(result)


if __name__ == "__main__":
    main()