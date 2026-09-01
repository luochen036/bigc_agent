from functools import lru_cache
from pathlib import Path

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


def main() -> None:
    model = load_model()

    vec = model.encode("北京印刷学院创办于哪一年？", normalize_embeddings=True)
    print(vec.shape)  # (512,) —— 一句话 → 512 维向量
    print(vec[:5])  # 看一眼向量长什么样


if __name__ == "__main__":
    main()
