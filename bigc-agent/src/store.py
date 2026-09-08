"""向量存储：用 Chroma 将原文、来源和向量一起保存在本地"""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

# 数据库路径同样基于代码文件位置定位，避免启动目录影响数据位置。
DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "chroma"
COLLECTION_NAME = "bigc_docs"


def get_collection(db_path: str | Path = DB_PATH) -> Collection:
    # PersistentClient 表示持久化客户端：程序退出后，数据仍保存在硬盘上。
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        # 告诉数据库用余弦距离比较向量；这和 retriever 中的 1 - distance 对应。
        metadata={"hnsw:space": "cosine"},
    )


def index_docs(
    chunks: list[str],
    sources: list[str],
    embeddings: list[list[float]],
    db_path: str | Path = DB_PATH,
) -> Collection:
    """将当前全部文档的文本块写入集合。三个列表的相同下标表示同一个块。"""
    if not (len(chunks) == len(sources) == len(embeddings)):
        raise ValueError("文本块、来源和向量的数量必须相同")
    if not chunks:
        raise ValueError("没有可入库的文本块，保留原索引")

    collection = get_collection(db_path)

    # 每个块需要唯一编号。:06d 表示把整数补零到 6 位，例如 c000000、c000001。
    ids = []
    metadatas = []
    for i, source in enumerate(sources):
        ids.append(f"c{i:06d}")
        metadatas.append({"source": source})

    # set 是集合，可用减法求出两个编号集合的差集。
    # 先记住旧编号，写入完成后才能判断哪些旧块已不在当前文档中。
    previous_ids = set(collection.get(include=[])["ids"])
    # upsert：编号已存在就更新，不存在就新增，所以重复运行不会无限重复插入。
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    # 文档减少时，清理多余旧块，防止已经删除的资料仍被检索到。
    # 这不是增量导入接口：调用者每次应传入当前“全部文档”的块。
    stale_ids = previous_ids - set(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    print(f"[OK] 已写入 {collection.count()} 条")
    return collection
