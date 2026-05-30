import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.faiss_retriever import FaissRetriever, build_embeddings

LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
MAX_LENGTH = 128
HF_FINE_TUNED_MODEL = "sovrncrypt/indobert-smsa-nolimit-ds-test"


def has_model_weights(path: Path) -> bool:
    return any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "tf_model.h5"))


FINE_TUNED_MODEL_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "indobert-sm-sa-notebook",
    PROJECT_ROOT / "outputs" / "indobert-sm-sa",
]
DEFAULT_MODEL_CHECKPOINT = next(
    (str(path) for path in FINE_TUNED_MODEL_CANDIDATES if path.exists() and has_model_weights(path)),
    HF_FINE_TUNED_MODEL,
)

st.set_page_config(page_title="IndoBERT Sentiment + FAISS", layout="wide")

st.title("IndoBERT Sentiment Analysis + Similarity Search (FAISS)")

with st.sidebar:
    st.header("Settings")
    train_csv = st.text_input("Train CSV", value="data/train.csv")
    embedder_checkpoint = st.text_input(
        "Embedding model", value="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
    max_retrieval_rows = st.number_input(
        "Max retrieval rows",
        min_value=100,
        max_value=11000,
        value=2000,
        step=100,
        help="Use fewer rows for faster local demos. Set to 11000 to index the full training set.",
    )
    model_checkpoint = st.text_input("Classifier model", value=DEFAULT_MODEL_CHECKPOINT)

if model_checkpoint == "indobenchmark/indobert-base-p1":
    st.warning(
        "You are using the base IndoBERT checkpoint. For sentiment predictions, use a fine-tuned "
        f"model such as {HF_FINE_TUNED_MODEL}."
    )
elif Path(model_checkpoint).exists() and not has_model_weights(Path(model_checkpoint)):
    st.error(
        "The selected model directory exists but does not contain model weights. "
        "Download/copy model.safetensors or pytorch_model.bin from the training output folder."
    )

query = st.text_area("Input text", height=120, placeholder="Ketik komentar/teks bahasa Indonesia...")

@st.cache_resource(show_spinner=False)
def load_classifier(model_checkpoint: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model_path = Path(model_checkpoint)
    if model_path.exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    model.eval()
    model.to(device)
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_embedder(embedder_checkpoint: str):
    return SentenceTransformer(embedder_checkpoint)


@st.cache_resource(show_spinner=True)
def build_retriever(train_csv: str, embedder_checkpoint: str, max_rows: int):
    df = pd.read_csv(train_csv)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    texts = df["text"].astype(str).tolist()

    labels = None
    if "label" in df.columns:
        # if labels are strings, map them
        if isinstance(df["label"].iloc[0], str):
            label_map = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
            labels = [label_map.get(x, -1) for x in df["label"].tolist()]
        else:
            labels = df["label"].tolist()

    embedder = load_embedder(embedder_checkpoint)
    embs = build_embeddings(texts, embedder, batch_size=32, normalize_embeddings=True)
    retriever = FaissRetriever(embs, texts=texts, labels=labels, metric="cosine")
    return retriever

if st.button("Predict & Retrieve") and query.strip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_classifier(model_checkpoint, device)

    # classification
    inputs = tokenizer(query, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = LABELS[pred_id]

    st.subheader("Classification")
    st.success(f"Predicted sentiment: **{pred_label}** | confidence: **{probs[pred_id]:.4f}**")
    st.dataframe(
        pd.DataFrame(
            [{"label": LABELS[i], "score": float(probs[i])} for i in range(len(LABELS))]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # retrieval
    with st.spinner(f"Building FAISS retriever from up to {max_retrieval_rows} training rows..."):
        retriever = build_retriever(train_csv, embedder_checkpoint, int(max_retrieval_rows))
    embedder = load_embedder(embedder_checkpoint)
    q_emb = build_embeddings([query], embedder, batch_size=8, normalize_embeddings=True)
    result = retriever.search(q_emb, top_k=top_k)
    items = retriever.get_items(result)

    st.subheader("Top-k Similar Training Examples")
    for it in items:
        lbl_id = it.get("label")
        lbl_str = LABELS.get(int(lbl_id), "N/A") if lbl_id is not None else "N/A"
        st.write(
            f"- score: {it['score']:.4f} | label: {lbl_str} | text: {it['text'][:180]}{'...' if len(it['text'])>180 else ''}"
        )

