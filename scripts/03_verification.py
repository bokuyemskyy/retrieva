import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load the model
# Using a popular, fast model. You can swap this for 'BAAI/bge-small-en-v1.5' later.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load Chunks & Create FAISS Index
chunks = []
with open("wikipedia_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Embedding {len(chunks)} chunks...")
chunk_texts = [c["content"] for c in chunks]
chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)

# Create the FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean) is standard
index.add(np.array(chunk_embeddings).astype("float32"))

# 3. Load Test Questions
with open("test_questions.json", "r", encoding="utf-8") as f:
    test_data = json.get(f) if hasattr(json, "get") else json.load(f)

# 4. Run Benchmark
hits = 0
top_k = 1

print(f"Testing {len(test_data)} questions...")

for item in test_data:
    query = item["question"]
    # We want to find the 'article_id' that this question came from
    target_id = item["ground_truth_id"]

    # Embed query and search
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    # Get the article_ids of the top 5 results
    retrieved_article_ids = [
        chunks[idx]["metadata"]["article_id"] for idx in indices[0]
    ]

    if target_id in retrieved_article_ids:
        hits += 1

# 5. Final Result
hit_rate = (hits / len(test_data)) * 100
print("\nBenchmark Results:")
print(f"Total Questions: {len(test_data)}")
print(f"Hit@{top_k}: {hit_rate:.2f}%")
