class RerankingRetriever:
    def __init__(self, base_retriever, reranker):
        self.base_retriever = base_retriever
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 30):
        candidates = self.base_retriever.retrieve(query, top_k=fetch_k)

        reranked = self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=top_k,
        )

        return reranked
