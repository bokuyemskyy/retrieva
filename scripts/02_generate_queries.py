import json
import random
from openai import OpenAI

client = OpenAI(
    api_key="sk-d7bf77d3973646878ee75fbdc38cc83c", base_url="https://api.deepseek.com"
)

all_chunks = []
with open("wikipedia_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        all_chunks.append(json.loads(line))

sample_chunks = random.sample(all_chunks, 50)

eval_dataset = []

print(f"Generating questions for {len(sample_chunks)} chunks...")

for chunk in sample_chunks:
    context = chunk["content"]

    prompt = f"""
    You are an expert at creating test datasets for information retrieval.
    Given the text below, generate 3 distinct questions that can be answered specifically by this text.
    
    Rules:
    1. The question must be answerable ONLY using the provided text.
    2. Do not use phrases like "According to the text".
    3. Make the questions natural, as a human would type them into a search engine.

    Text: {context}
    
    Provide the output in a JSON list of strings only.
    Example: ["What is a qubit?", "How is a qubit physically realized?"]
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            stream=False,
        )

        questions = json.loads(response.choices[0].message.content)
        # q_list = questions.get("questions", list(questions.values())[0])

        for q in questions:
            eval_dataset.append(
                {
                    "question": q,
                    "ground_truth_id": chunk["metadata"]["article_id"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                }
            )

    except Exception as e:
        print(f"Error generating for chunk: {e}")

with open("test_questions.json", "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=4)

print(f"Saved {len(eval_dataset)} test questions to test_questions.json")
