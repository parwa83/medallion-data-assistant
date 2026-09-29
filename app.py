"""Standalone Streamlit front end that calls the existing rag_gold.py CLI."""
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="RAG : Query about Gold Layers", layout="wide")
APP_DIR = Path(__file__).resolve().parent
RAG_SCRIPT = APP_DIR / "rag_gold.py"


def get_hf_token() -> str | None:
    try:
        secret = st.secrets.get("HF_TOKEN")
    except Exception:
        secret = None
    if secret:
        return str(secret)
    if os.getenv("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        return os.environ["HUGGINGFACEHUB_API_TOKEN"]
    # Supports the local key module some users already use with rag_gold.py.
    try:
        from api_key import HF_TOKEN
        return str(HF_TOKEN)
    except (ImportError, AttributeError):
        return None


def run_rag_cli(args: list[str], token: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    if not RAG_SCRIPT.is_file():
        raise FileNotFoundError(f"Could not find {RAG_SCRIPT}. Keep streamlit_app.py beside rag_gold.py.")
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    return subprocess.run(
        [sys.executable, str(RAG_SCRIPT), *args],
        cwd=APP_DIR,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


st.title("RAG : Query about Gold Layers")
st.caption("Ask questions about following gold layers  customer_360 ||  daily_revenue_metrics ||  product_performance ||  store_performance")

with st.sidebar:
    st.header("Settings")
    output_root = st.text_input("Medallion output folder", "data/lake")
    embedding_model = st.text_input(
        "Embedding model", os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chat_model = st.text_input(
        "Chat model", os.getenv("HF_CHAT_MODEL", "openai/gpt-oss-120b:fastest")
    )
    top_k = st.slider("Rows to retrieve", min_value=1, max_value=20, value=6)

output_path = Path(output_root).expanduser()
if not output_path.is_absolute():
    output_path = APP_DIR / output_path
index_path = output_path / "rag" / "gold_index.jsonl"
gold_names = ("daily_revenue_metrics", "customer_360", "product_performance", "store_performance")
missing_gold = [name for name in gold_names if not (output_path / "gold" / name).exists()]

# if not RAG_SCRIPT.exists():
#     st.error("rag_gold.py is missing. Put this UI file beside your existing rag_gold.py.")
# elif missing_gold:
#     st.warning("Gold tables not found: " + ", ".join(missing_gold) + ". Run the medallion pipeline first.")
# elif index_path.exists():
#     st.success(f"Gold tables and vector index found: `{index_path}`")
# else:
#     st.info("Gold tables found. Build the vector index to start asking questions.")

token = get_hf_token()
if not token:
    st.warning("Hugging Face token not found. Set HF_TOKEN in this terminal or in .streamlit/secrets.toml.")

build_disabled = bool(missing_gold) or not RAG_SCRIPT.exists() or not token
# if st.button("Build / refresh Gold index", disabled=build_disabled):
#     with st.spinner("Building the Gold-only vector index…"):
#         try:
#             result = run_rag_cli(
#                 ["--output", str(output_path), "--embedding-model", embedding_model, "build"], token
#             )
#             if result.stdout:
#                 with st.expander("Index build output", expanded=True):
#                     st.code(result.stdout)
#             if result.returncode:
#                 st.error(result.stderr or "Index build failed. See output above.")
#             else:
#                 st.success("Gold vector index built.")
#                 st.rerun()
#         except Exception as exc:
#             st.error(f"Could not build the index: {exc}")

st.divider()
st.subheader("Ask a question")
question = st.text_input("Question", placeholder="Which stores had the highest revenue?")
if st.button("Ask", type="primary", disabled=not question.strip() or not index_path.exists() or not token):
    with st.spinner("Searching Gold records and generating an answer…"):
        try:
            result = run_rag_cli(
                ["--output", str(output_path), "--embedding-model", embedding_model,
                 "--chat-model", chat_model, "ask", question.strip(), "--top-k", str(top_k)], token
            )
            if result.returncode:
                st.error(result.stderr or "The RAG command failed.")
                if result.stdout:
                    st.code(result.stdout)
            else:
                output = result.stdout.strip()
                answer_marker = "Answer\n------\n"
                records_marker = "\nRetrieved records\n-----------------\n"
                if answer_marker in output:
                    answer_text = output.split(answer_marker, 1)[1]
                    answer, _, records = answer_text.partition(records_marker)
                    st.markdown("### Answer")
                    st.write(answer.strip())
                    if records.strip():
                        with st.expander("Retrieved Gold records", expanded=True):
                            st.code(records.strip())
                else:
                    st.code(output or "The RAG script returned no output.")
        except Exception as exc:
            st.error(f"Could not run the RAG pipeline: {exc}")

st.caption("The existing rag_gold.py handles indexing and retrieval. Hugging Face receives the question and retrieved Gold row text for inference.")
