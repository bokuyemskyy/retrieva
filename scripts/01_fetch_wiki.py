import json
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True, split="train")

shuffled_ds = ds.shuffle(seed=42, buffer_size=1000)

articles = []
for i, entry in enumerate(shuffled_ds):
    if i >= 100:
        break
    articles.append({"id": entry["id"], "title": entry["title"], "text": entry["text"]})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

all_chunks = []

for article in articles:
    chunks = text_splitter.split_text(article["text"])

    for i, chunk_text in enumerate(chunks):
        all_chunks.append(
            {
                "metadata": {
                    "article_id": article["id"],
                    "title": article["title"],
                    "chunk_index": i,
                },
                "content": chunk_text,
            }
        )

with open("wikipedia_chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")

print(f"Saved {len(all_chunks)} chunks to wikipedia_chunks.jsonl")
