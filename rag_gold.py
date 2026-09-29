"""Build and query a small local vector index from Gold Delta tables only.

Hugging Face Inference API is used for embeddings and answer generation.
Set HF_TOKEN in the environment; never paste the token into this file.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
from huggingface_hub import InferenceClient

from medallion_local import make_spark


GOLD_TABLES = (
    "daily_revenue_metrics",
    "customer_360",
    "product_performance",
    "store_performance",
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def row_to_text(table: str, row: dict[str, Any]) -> str:
    """Create a searchable, labeled text representation from one Gold row."""
    fields = [f"Gold table: {table}"]
    for key, value in row.items():
        if key.startswith("_") or value is None:
            continue
        fields.append(f"{key}: {value}")
    return " | ".join(fields)


def vectorize(client: InferenceClient, model: str, texts: list[str]) -> np.ndarray:
    """Request vectors and mean-pool token embeddings when the provider returns them."""
    output = np.asarray(client.feature_extraction(texts, model=model), dtype=np.float32)
    if output.ndim == 3:  # batch x tokens x dimensions
        output = output.mean(axis=1)
    elif output.ndim == 1:
        output = output.reshape(1, -1)
    if output.ndim != 2 or output.shape[0] != len(texts):
        raise ValueError(f"Unexpected embedding result shape: {output.shape}")
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return output / np.maximum(norms, 1e-12)


def build_index(output_dir: Path, index_file: Path, token: str,
                embedding_model: str, batch_size: int) -> None:
    missing = [name for name in GOLD_TABLES if not (output_dir / "gold" / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Gold Delta tables are missing: " + ", ".join(missing)
            + f". Run medallion_local.py first (expected under {output_dir / 'gold'})."
        )

    client = InferenceClient(token=token)
    spark = make_spark(output_dir / "_warehouse")
    entries: list[dict[str, Any]] = []
    try:
        for table in GOLD_TABLES:
            path = output_dir / "gold" / table
            frame = spark.read.format("delta").load(str(path))
            rows = frame.drop("_gold_processed_timestamp").toLocalIterator()
            count = 0
            for row in rows:
                payload = row.asDict(recursive=True)
                entries.append({"table": table, "row": payload, "text": row_to_text(table, payload)})
                count += 1
            print(f"Loaded {count:,} Gold rows from {table}")
    finally:
        spark.stop()

    if not entries:
        raise ValueError("Gold tables contain no rows to index.")

    index_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = index_file.with_suffix(index_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as destination:
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            vectors = vectorize(client, embedding_model, [item["text"] for item in batch])
            for item, vector in zip(batch, vectors):
                item["embedding"] = vector.tolist()
                destination.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            print(f"Embedded {min(start + len(batch), len(entries)):,}/{len(entries):,} rows")
    temp_file.replace(index_file)
    print(f"Saved {len(entries):,} vectors to {index_file}")


def load_index(index_file: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not index_file.exists():
        raise FileNotFoundError(f"Vector index not found: {index_file}. Run the build command first.")
    records = []
    with index_file.open(encoding="utf-8") as source:
        for line in source:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Vector index is empty: {index_file}")
    vectors = np.asarray([record["embedding"] for record in records], dtype=np.float32)
    return records, vectors


def select_retrieval(question: str, records: list[dict[str, Any]], vectors: np.ndarray,
                     query_vector: np.ndarray, top_k: int) -> tuple[list[int], np.ndarray, str]:
    """Route entity-specific questions to the right Gold table and honor superlatives."""
    q = question.lower()
    table = None
    if re.search(r"\b(store|stores|branch|branches|location|locations)\b", q):
        table = "store_performance"
    elif re.search(r"\b(product|products|sku|category|categories|item|items)\b", q):
        table = "product_performance"
    elif re.search(r"\b(customer|customers|buyer|buyers|lifetime|payment method)\b", q):
        table = "customer_360"
    elif re.search(r"\b(daily|day|days|date|cancel rate|delivery rate)\b", q):
        table = "daily_revenue_metrics"

    candidate_ids = [i for i, item in enumerate(records) if table is None or item["table"] == table]
    if not candidate_ids:
        candidate_ids = list(range(len(records)))

    metric = None
    if re.search(r"\b(revenue|sales|earnings)\b", q):
        metric = "total_revenue"
    elif re.search(r"\b(order value|average order|aov)\b", q):
        metric = "avg_order_value"
    elif re.search(r"\b(cancel|cancellation)\b", q):
        metric = "cancel_rate_pct" if "cancel_rate_pct" in records[candidate_ids[0]]["row"] else "cancelled_orders"
    elif re.search(r"\b(quantity|units sold)\b", q):
        metric = "total_quantity_sold"
    elif re.search(r"\b(spend|spent|lifetime)\b", q):
        metric = "total_lifetime_spend"

    high = bool(re.search(r"\b(highest|top|most|best|largest|greatest)\b", q))
    low = bool(re.search(r"\b(lowest|bottom|least|worst|smallest)\b", q))
    if metric and (high or low) and all(metric in records[i]["row"] for i in candidate_ids):
        candidate_ids = [i for i in candidate_ids if records[i]["row"].get(metric) is not None]
        candidate_ids.sort(
            key=lambda i: float(records[i]["row"].get(metric) or 0),
            reverse=high,
        )
        chosen = candidate_ids[:top_k]
        scores = np.asarray([float(records[i]["row"].get(metric) or 0) for i in chosen], dtype=np.float32)
        return chosen, scores, metric

    cosine_scores = vectors @ query_vector
    chosen = sorted(candidate_ids, key=lambda i: cosine_scores[i], reverse=True)[:top_k]
    return chosen, cosine_scores, "similarity"


def answer_question(question: str, index_file: Path, token: str,
                    embedding_model: str, chat_model: str, top_k: int) -> None:
    client = InferenceClient(token=token)
    records, vectors = load_index(index_file)
    query_vector = vectorize(client, embedding_model, [question])[0]
    best, scores, score_label = select_retrieval(question, records, vectors, query_vector, top_k)
    evidence = "\n".join(
        f"[{records[i]['table']} | {score_label}={scores[position]:.3f}] {records[i]['text']}"
        for position, i in enumerate(best)
    )
    response = client.chat_completion(
        model=chat_model,
        messages=[
            {"role": "system", "content": (
                "Answer using only the supplied retrieved Gold-layer records. "
                "If they do not contain enough information, say so. Distinguish exact values "
                "from estimates and cite the Gold table names used. Do not invent facts."
            )},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved Gold records:\n{evidence}"},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    print("\nAnswer\n------")
    print(response.choices[0].message.content)
    print("\nRetrieved records")
    print("-----------------")
    for position, i in enumerate(best):
        print(f"[{records[i]['table']}] {score_label}={scores[position]:.3f}: {records[i]['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/lake"), help="Medallion output root")
    parser.add_argument("--index", type=Path, default=None, help="JSONL vector index path")
    parser.add_argument("--embedding-model", default=os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--chat-model", default=os.getenv("HF_CHAT_MODEL", DEFAULT_CHAT_MODEL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Embed all rows from Gold tables")
    build.add_argument("--batch-size", type=int, default=16)
    ask = subparsers.add_parser("ask", help="Retrieve Gold rows and ask a question")
    ask.add_argument("question", help="Question about the Gold-layer data")
    ask.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        parser.error("Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in your environment first.")
    output_dir = args.output.expanduser().resolve()
    index_file = args.index.expanduser().resolve() if args.index else output_dir / "rag" / "gold_index.jsonl"
    if args.command == "build":
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        build_index(output_dir, index_file, token, args.embedding_model, args.batch_size)
    else:
        if args.top_k < 1:
            parser.error("--top-k must be at least 1")
        answer_question(args.question, index_file, token, args.embedding_model, args.chat_model, args.top_k)


if __name__ == "__main__":
    main()
