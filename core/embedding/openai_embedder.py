from typing import List

from core.embedding.base_embedder import BaseEmbedder
from core.model.chunk import Chunk


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        client,
        model: str = "text-embedding-3-small",
        vector_size: int = 1536,
    ) -> None:
        self._client = client
        self._model = model
        self.vector_size = vector_size

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks

        texts = [c.content.replace("\n", " ").strip() for c in chunks]
        response = self._client.embeddings.create(input=texts, model=self._model)

        for chunk, item in zip(chunks, response.data):
            chunk.embedding = item.embedding

        return chunks
