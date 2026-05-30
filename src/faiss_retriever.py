import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "faiss is required. Install with: pip install faiss-cpu"
    ) from e


@dataclass
class RetrievalResult:
    indices: List[int]
    scores: List[float]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


class FaissRetriever:
    """A small FAISS wrapper for cosine-similarity retrieval."""

    def __init__(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        metric: str = "cosine",
    ):
        if metric not in {"cosine"}:
            raise ValueError("Only metric='cosine' is supported in this implementation.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.metric = metric
        self.texts = texts
        self.labels = labels

        # cosine -> normalize embeddings and use inner product
        self.embeddings = _l2_normalize(embeddings)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> RetrievalResult:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")

        query_embeddings = _l2_normalize(query_embeddings)
        scores, indices = self.index.search(query_embeddings, top_k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        return RetrievalResult(indices=indices, scores=scores)

    def get_items(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for idx, score in zip(result.indices, result.scores):
            item = {
                "index": int(idx),
                "score": float(score),
                "text": self.texts[idx],
            }
            if self.labels is not None:
                item["label"] = self.labels[idx]
            items.append(item)
        return items


def build_embeddings(
    texts: List[str],
    embedder: Any,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Uses a sentence-transformers-like embedder with encode()."""

    # sentence-transformers returns np.ndarray already
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    if not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    if emb.dtype != np.float32:
        emb = emb.astype("float32")
    return emb


def save_faiss_index(index: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> Any:
    return faiss.read_index(path)

