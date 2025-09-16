import json
import pandas as pd
from pathlib import Path

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain.chains import ConversationalRetrievalChain

# --- your project code ---
from RAGAgentCloud.llm import llm
from RAGAgentCloud.retriever import get_collection_retriever
from RAGAgentCloud.prompt import prompt, condense_prompt

TESTSET_PATH = Path("testset.json")
CSV_OUT = Path("ragas_results.csv")
JSON_OUT = Path("ragas_results.json")
SUMMARY_OUT = Path("ragas_summary.json")

# 1) Load strict testset
if not TESTSET_PATH.exists():
    raise SystemExit("testset.json not found. Run dataset_generator.py first.")
with TESTSET_PATH.open("r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 2) Set up RAG chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=get_collection_retriever(),
    condense_question_prompt=condense_prompt,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# 3) Helper to run the chain
def run_rag(question: str):
    result = rag_chain({"question": question, "chat_history": []})
    answer = result.get("answer") or result.get("result") or ""
    contexts = [doc.page_content for doc in result.get("source_documents", [])]
    return answer, contexts

# 4) Build RAGAS dataset (answers/contexts from your chain)
dataset = []
for item in eval_data:
    q = item["question"]
    gt = item["ground_truth"]
    answer, contexts = run_rag(q)
    dataset.append(
        {
            "question": q,
            "answer": answer,
            "ground_truth": gt,
            "contexts": contexts,
        }
    )

# 5) Evaluate
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
results = evaluate(dataset, metrics=metrics)

# 6) Normalize results → DataFrame
def results_to_dataframe(res_obj, fallback_dataset):
    """
    Supports different ragas versions. Prefers .to_pandas() if present.
    Falls back to constructing a DataFrame from dict-like structures.
    """
    # Preferred API
    if hasattr(res_obj, "to_pandas"):
        df = res_obj.to_pandas()
        # Try to ensure we include question text; if not, stitch it in
        if "question" not in df.columns:
            questions = [d["question"] for d in fallback_dataset]
            df.insert(0, "question", questions)
        return df

    # Dict-like fallback (metric_name -> list of scores)
    if isinstance(res_obj, dict):
        # expect something like {"faithfulness": [...], "answer_relevancy": [...], ...}
        df = pd.DataFrame(res_obj)
        questions = [d["question"] for d in fallback_dataset]
        if len(df) == len(questions) and "question" not in df.columns:
            df.insert(0, "question", questions)
        return df

    # Last resort: single-row summary (means)
    try:
        return pd.DataFrame([res_obj])  # best-effort
    except Exception:
        raise RuntimeError("Unsupported ragas results format; please check your ragas version.")

df = results_to_dataframe(results, dataset)

# 7) Persist per-sample results
df.to_csv(CSV_OUT, index=False)
df.to_json(JSON_OUT, orient="records", force_ascii=False, indent=2)

# 8) Compute & persist aggregate means
numeric_cols = [c for c in df.columns if c != "question"]
means = {c: float(df[c].mean()) for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])}
with SUMMARY_OUT.open("w", encoding="utf-8") as f:
    json.dump({"metric_means": means}, f, ensure_ascii=False, indent=2)

print("\n==== RAGAS Evaluation ====")
print(df.head())
print("\nMetric means:")
for k, v in means.items():
    print(f"  {k}: {v:.4f}")
print(f"\nSaved per-sample: {CSV_OUT}, {JSON_OUT}")
print(f"Saved summary:    {SUMMARY_OUT}")