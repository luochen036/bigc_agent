"""资料检索：输入一个问题，返回语义最相近的几个文本块。"""


def retrieve(collection, query: str, embedder, k: int = 3, min_score: float = 0.3) -> list[dict]:
    """返回 [{"text": 原文, "source": 文件名, "score": 相似度}, ...]。
    collection 是数据库集合，embedder 是本地嵌入模型。
    k 表示最多取几个块，min_score 是最低相似度。0.3 只是教学起点，并非通用标准。
    """
    if not query.strip():
        raise ValueError("问题不能为空")
    if k < 1:
        raise ValueError("k 必须大于 0")
    if not -1 <= min_score <= 1:
        raise ValueError("min_score 必须在 -1 到 1 之间")
    count = collection.count()
    if count == 0:
        return []
    # encode 接受一批文本，所以单个问题也写成 [query]。
    # 返回的也是一批向量，[0] 取出唯一那个问题的向量。
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    hits = collection.query(
        query_embeddings=[q_vec.tolist()],
        # 数据库只有 2 条时不能硬取 3 条，min() 取两者较小值。
        n_results=min(k, count),
    )

    # Chroma 支持同时查询多个问题，返回值因此多包了一层列表。
    # 我们只查一个问题，三个 [0] 都是在取“第一个问题的全部命中结果”。
    docs = hits["documents"][0]
    metas = hits["metadatas"][0]
    dists = hits["distances"][0]

    results = []
    # zip 将三个列表按位置配对：同一个块的原文、来源和距离一起处理。
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        if score < min_score:
            continue
        results.append({
            "text": doc,
            # get 的第二个参数是缺省值；没有来源记录时显示问号。
            "source": (meta or {}).get("source", "?"),
            "score": score,
        })
    return results
