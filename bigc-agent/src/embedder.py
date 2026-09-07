from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "BAAI--bge-small-zh-v1.5"
    / "snapshots"
    / "master"
)


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load the local model snapshot when available, otherwise use Hugging Face."""
    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else MODEL_NAME
    return SentenceTransformer(model_source)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    return float(np.dot(a, b) / denominator)


def main() -> None:
    model = load_model()

    v1, v2, v3, v4 = model.encode(
        [
            "印刷工程专业怎么样？",
            "学校有哪些特色专业？",
            "今天天气不错。",
            "超人",
        ],
        normalize_embeddings=True,
    )
    print(v1)
    print("印刷 vs 特色专业:", round(cosine(v1, v2), 3))
    print("印刷 vs 天气     :", round(cosine(v1, v3), 3))
    print("印刷 vs 超人     :", round(cosine(v1, v4), 3))


if __name__ == "__main__":
    main()
