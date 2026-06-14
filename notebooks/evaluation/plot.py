import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models_files = {
    "Sparse search": "results/ragas_text_means.csv",
    "nomic-embed-text": "results/ragas_nomic_means.csv",
    "text-embeddings-3-large": "results/ragas_large_means.csv",
    "qwen3-embedding:8b": "results/ragas_qwen_means.csv",
    "text-embeddings-3-large\nwith reranker": "results/ragas_reranker_means1.csv",
}

metrics = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]

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
        print(f"Warning: {file_name} not found")

df_plot = pd.DataFrame(data)

if not df_plot.empty:
    x = np.arange(len(df_plot))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ["#5BA4CF", "#4785A8", "#346682", "#21475C"]

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width

        ax.bar(
            x + offset,
            df_plot[metric],
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
            edgecolor="white",
        )

    ax.set_ylabel("Score")
    ax.set_xlabel("Retrieval configuration")

    ax.set_xticks(x)
    ax.set_xticklabels(
        df_plot["model"],
        rotation=12,
        ha="right",
    )

    ax.set_ylim(0, 1)

    ax.legend(
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    plt.tight_layout()

    plt.savefig(
        "metrics_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

else:
    print("No data loaded. Please check your CSV files.")
