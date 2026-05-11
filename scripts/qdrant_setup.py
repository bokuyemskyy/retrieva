from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

load_dotenv()

COLLECTION_NAME = "documents"

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

client = QdrantClient(url=os.getenv("QDRANT_URL"))

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print("Qdrant ready with BGE-M3.")
