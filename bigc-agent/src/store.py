from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

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
