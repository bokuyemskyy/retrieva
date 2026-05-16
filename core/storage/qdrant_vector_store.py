from typing import List, Dict, Any

from core.model.chunk import Chunk


class QdrantVectorStore:
    """Manages the direct connection to Qdrant/Qdroid."""

    def __init__(self, connection_string: str):
        self.client = None  # Initialize Qdrant Client here

    def upsert_chunks(self, chunks: List[Chunk]):
        # Store both dense vectors and sparse (BM25) tokens into Qdrant payload/vectors
        pass

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        metadata_filter: Dict[str, Any],
        top_k: int,
    ) -> List[Chunk]:
        # Uses Qdrant's internal Prefetch / Query API to combine
        # Dense and BM25 search with native metadata filtering
        return []
