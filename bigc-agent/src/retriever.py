# src/retriever.py
def retrieve(collection, query: str, embedder, k: int = 3, min_score: float = 0.0) -> list[dict]:
    """向量检索：query → 向量 → top-k 相似文本。"""
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    hits = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
    )
    docs, metas, dists = hits["documents"][0], hits["metadatas"][0], hits["distances"][0]

    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist                    # 余弦距离 → 相似度
        if score < min_score:
            continue                        # 低于阈值：资料无关，宁可拒答
        results.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "score": score,
        })
    return results