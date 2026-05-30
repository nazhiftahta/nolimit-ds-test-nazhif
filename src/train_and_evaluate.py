import argparse
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.utils import logging as transformers_logging

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.faiss_retriever import FaissRetriever, build_embeddings


LABELS = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
LABEL2ID = {v: k for k, v in LABELS.items()}
MAX_LENGTH = 128


def configure_runtime_output() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*`tokenizer` is deprecated and will be removed.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*'pin_memory' argument is set as true but no accelerator is found.*",
        category=UserWarning,
    )
    transformers_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed") or col == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # expected columns: text,label (label optional for inference)
    if "text" not in df.columns:
        raise ValueError(f"CSV at {path} must include 'text' column")
    df["text"] = df["text"].astype(str)

    if "label" in df.columns:
        # ensure numeric labels
        if df["label"].dtype == object:
            df["label"] = df["label"].map(LABEL2ID)
        df["label"] = df["label"].astype("int64")
    return Dataset.from_pandas(df, preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def tokenize_dataset(tokenizer, ds: Dataset) -> Dataset:
    def preprocess(examples: Dict[str, List[Any]]):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    return ds.map(preprocess, batched=True)


def build_training_arguments(**kwargs) -> TrainingArguments:
    """Support both old and new Transformers strategy argument names."""

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def train(
    train_csv: str,
    validation_csv: str,
    model_checkpoint: str,
    output_dir: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    seed: int,
    max_train_samples: int,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:

    set_seed(seed)

    train_ds = load_csv_dataset(train_csv)
    val_ds = load_csv_dataset(validation_csv)

    if max_train_samples and max_train_samples > 0:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_tok = tokenize_dataset(tokenizer, train_ds)
    val_tok = tokenize_dataset(tokenizer, val_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=3,
        id2label=LABELS,
        label2id={"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = build_training_arguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        seed=seed,
        report_to=[],
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, csv_path: str, batch_size: int = 16):
    ds = load_csv_dataset(csv_path)
    tok = tokenize_dataset(tokenizer, ds)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    pred = trainer.predict(tok)
    logits = pred.predictions
    preds = np.argmax(logits, axis=-1)

    if "label" in ds.column_names:
        labels = np.array(ds["label"], dtype=int)
        print("Accuracy:", accuracy_score(labels, preds))
        print("F1 weighted:", f1_score(labels, preds, average="weighted"))
        print(classification_report(labels, preds, target_names=[LABELS[i] for i in range(3)]))

    return ds, preds


def run_retrieval(
    train_csv: str,
    model_checkpoint: str,
    embedder_checkpoint: str,
    sample_input_csv: str,
    top_k: int,
    seed: int,
) -> None:
    # embeddings come from sentence-transformers
    embedder = SentenceTransformer(embedder_checkpoint)

    train_df = pd.read_csv(train_csv)
    texts = train_df["text"].astype(str).tolist()
    if "label" in train_df.columns:
        labels = train_df["label"].tolist()
        if isinstance(labels[0], str):
            labels = [LABEL2ID[x] for x in labels]
    else:
        labels = None

    # Build index over TRAIN texts
    train_embeddings = build_embeddings(texts, embedder, batch_size=32, normalize_embeddings=True)
    retriever = FaissRetriever(train_embeddings, texts=texts, labels=labels, metric="cosine")

    sample_df = pd.read_csv(sample_input_csv)
    sample_texts = sample_df["text"].astype(str).tolist()

    query_embeddings = build_embeddings(sample_texts, embedder, batch_size=8, normalize_embeddings=True)

    for i, qt in enumerate(sample_texts):
        result = retriever.search(query_embeddings[i : i + 1], top_k=top_k)
        items = retriever.get_items(result)
        print("\n=== Query ===")
        print(qt)
        print("=== Top-k similar examples ===")
        for it in items:
            lbl = it.get("label", None)
            lbl_str = LABELS[lbl] if lbl is not None and int(lbl) in LABELS else lbl
            print(f"score={it['score']:.4f} label={lbl_str} text={it['text'][:120]}")


def main():
    configure_runtime_output()

    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--validation_csv", type=str, default="data/validation.csv")
    p.add_argument("--test_csv", type=str, default="data/test.csv")
    p.add_argument("--sample_input_csv", type=str, default="data/sample_input.csv")

    p.add_argument("--model_checkpoint", type=str, default="indobenchmark/indobert-base-p1")
    p.add_argument("--embedder_checkpoint", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    p.add_argument("--output_dir", type=str, default="outputs/indobert-sm-sa")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=0)

    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--do_retrieval", action="store_true")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    tokenizer = None

    if args.do_train:
        model, tokenizer = train(
            train_csv=args.train_csv,
            validation_csv=args.validation_csv,
            model_checkpoint=args.model_checkpoint,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
        )
        model.to(device)
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir)

    if args.do_eval:
        if model is None or tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        predict(model.to(device), tokenizer, args.test_csv, batch_size=args.batch_size)

        # also show predictions on sample inputs (labels may be absent)
        if os.path.exists(args.sample_input_csv):
            df = pd.read_csv(args.sample_input_csv)
            if "label" not in df.columns:
                df["text"] = df["text"].astype(str)
                ds = Dataset.from_pandas(df)
                tok = tokenize_dataset(tokenizer, ds)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                trainer = Trainer(model=model.to(device), tokenizer=tokenizer, data_collator=data_collator)
                pred = trainer.predict(tok)
                probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
                preds = np.argmax(probs, axis=-1)
                print("\n=== Sample input classification ===")
                for t, pr, prob in zip(df["text"].astype(str).tolist(), preds.tolist(), probs.tolist()):
                    confidence = prob[int(pr)]
                    class_scores = ", ".join(f"{LABELS[i]}={prob[i]:.4f}" for i in range(len(LABELS)))
                    print(f"text={t[:80]}... => {LABELS[int(pr)]} | confidence={confidence:.4f} | {class_scores}")

    if args.do_retrieval:
        run_retrieval(
            train_csv=args.train_csv,
            model_checkpoint=args.model_checkpoint,
            embedder_checkpoint=args.embedder_checkpoint,
            sample_input_csv=args.sample_input_csv,
            top_k=args.top_k,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

