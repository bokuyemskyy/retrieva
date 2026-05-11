from pathlib import Path

from core.ingestion.file_ingestor import TextIngestor
from core.embeddings.local_embedding_model import LocalEmbeddingModel
from core.vectorstore.faiss_vector_store import FAISSVectorStore


def main():
    ingestor = TextIngestor(chunk_size=800, chunk_overlap=100)

    raw_dir = Path("data/raw/git_docs")
    files = list(raw_dir.glob("*.txt"))

    all_chunks = []

    for file_path in files:
        print(f"Processing: {file_path.name}")

        documents = ingestor.load(str(file_path))
        chunks = ingestor.chunk(documents)

        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    embedder = LocalEmbeddingModel()

    texts = [c.content for c in all_chunks]
    embeddings = embedder.embed_documents(texts)

    print(f"Embedding dimension: {len(embeddings[0])}")

    store = FAISSVectorStore(embedding_dim=len(embeddings[0]))

    store.add(
        embeddings, [{"content": c.content, "metadata": c.metadata} for c in all_chunks]
    )

    store.save()

    while True:
        query = input("\nEnter a query: ")

        if query == "exit":
            break

        query_embedding = embedder.embed_queries([query])[0]

        results = store.search(query_embedding, k=5)

        print("\nTop results:\n")

        for i, r in enumerate(results):
            print(f"[{i + 1}]")
            print(r["content"])
            print("META:", r["metadata"])
            print("SCORE:", r["score"])
            print("-" * 50)


if __name__ == "__main__":
    main()
