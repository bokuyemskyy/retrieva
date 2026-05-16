from core.retrieval.reranking.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, chunks: List[Chunk], top_n: int) -> List[Chunk]:
        # Implement cross-encoder scoring logic (e.g., Cohere, BGE-Reranker)
        # Sort chunks based on new scores and return top_n
        return chunks[:top_n]
