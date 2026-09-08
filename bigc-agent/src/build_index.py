"""第一个主入口：运行 python build_index.py，将文档保存为可检索的向量索引。

这是准备阶段。首次使用、增加文档、修改文档或删除文档后运行一次。
每次提问时不用重新入库，agent.py 会直接打开已有索引。
"""

from data_loader import DOCS_DIR, load_docs
from embedder import load_model
from splitter import split_text
from store import DB_PATH, index_docs


def build_index(data_dir=DOCS_DIR, db_path=DB_PATH):
    """读取文档 → 切块 → 向量化 → 写入数据库，返回数据库集合。"""
    # 第一步：docs 是字典列表，每项包含一篇文档的 source 和 text。
    # 参数有默认值，直接 build_index() 就能使用项目的文档和数据库目录。
    docs = load_docs(data_dir)
    chunks = []
    sources = []

    # 第二步：每篇长文切成多块。每个块都记录来源，回答时才能标注出处。
    # 两个列表的位置一一对应：chunks[0] 的来源是 sources[0]。
    for doc in docs:
        parts = split_text(doc["text"])
        for part in parts:
            chunks.append(part)
            sources.append(doc["source"])
    if not chunks:
        raise ValueError("没有找到非空 .txt 文档，保留原索引；请检查文档目录")

    print(f"[INFO] 已切分为 {len(chunks)} 个文本块，开始向量化")

    # 第三步：本地嵌入模型将每个文本块变成一组数字，即“向量”。
    # normalize_embeddings=True 将向量长度统一为 1，便于比较方向相似度。
    # encode 返回数组，tolist 将其转为数据库接口接受的 Python 列表。
    model = load_model()
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

    # 第四步：同时保存原文、来源和向量。仅存向量无法还原原文，因此三者都需要。
    collection = index_docs(chunks, sources, embeddings, db_path=db_path)
    print(f"[OK] 索引已保存到 {db_path}")
    return collection


def main() -> int:
    """使用固定默认目录，不需要传入命令行参数。"""
    try:
        build_index()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
