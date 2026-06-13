import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models_files = {
    "nomic-embed-text": "ragas_nomic_means.csv",
    "mxbai-embed-large": "ragas_mxbai_means.csv",
    "qwen3-embedding:8b": "ragas_qwen_means.csv",
    "text-embeddings-3-small": "ragas_small_means.csv",
    "text-embeddings-3-large": "ragas_large_means.csv",
}

metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

data = []
for model_name, file_name in models_files.items():
    try:
        df = pd.read_csv(file_name)
        row = df.iloc[0]
        data.append(
            {
                "model": model_name,
                "context_precision": row["context_precision"],
                "context_recall": row["context_recall"],
                "faithfulness": row["faithfulness"],
                "answer_relevancy": row["answer_relevancy"],
            }
        )
    except FileNotFoundError:
        print(
            f"Warning: {file_name} not found. Please ensure it is in the same directory."
        )

df_plot = pd.DataFrame(data)

if not df_plot.empty:
    x = np.arange(len(df_plot["model"]))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#5BA4CF", "#4785A8", "#346682", "#21475C"]

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            df_plot[metric],
            width,
            label=metric,
            color=colors[i],
            edgecolor="white",
        )

    ax.set_ylabel("Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["model"], rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    plt.savefig("metrics_comparison.png", dpi=300)
    plt.show()
else:
    print("No data loaded. Please check your CSV files.")
