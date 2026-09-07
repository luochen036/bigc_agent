from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from embedder import load_model
from data_loader import load_docs
from splitter import split_text



DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "chroma"
COLLECTION_NAME = "bigc_docs"


def get_collection(db_path: str | Path = DB_PATH) -> Collection:
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_docs(
    chunks: list[str],
    sources: list[str],
    embeddings: list[list[float]],
    db_path: str | Path = DB_PATH,
) -> Collection:
    if not (len(chunks) == len(sources) == len(embeddings)):
        raise ValueError("chunks, sources and embeddings must have the same length")

    collection = get_collection(db_path)
    collection.upsert(
        ids=[f"c{i:06d}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources],
    )
    print(f"[OK] 已写入 {collection.count()} 条")
    return collection


if __name__ == "__main__":
    docs = load_docs()
    model = load_model()

    all_chunks = []
    all_sources = []

    for doc in docs:
      chunks = split_text(doc["text"])
      all_chunks.extend(chunks)
      all_sources.extend([doc["source"]] * len(chunks))

    if not all_chunks:
        raise ValueError("没有找到可以写入数据库的文本块")

    all_embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,
    ).tolist()
    index_docs(
      all_chunks,
      all_sources,
      all_embeddings,
    )
    col = get_collection()
    print(f"当前向量数据库有 {col.count()} 条记录")
        