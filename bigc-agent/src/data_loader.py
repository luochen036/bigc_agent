from pathlib import Path


def load_docs(data_dir: str = "data/docs") -> list[dict]:
    """读取目录下所有 .txt 文档，返回 [{"source": 文件名, "text": 全文}]。"""
    docs = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        # 简单清洗：折叠连续空行、去掉首尾空白
        lines = [ln.strip() for ln in text.splitlines()]
        cleaned = "\n".join(ln for ln in lines if ln)
        docs.append({"source": path.name, "text": cleaned})
    print(f"[OK] 已加载 {len(docs)} 份文档")
    return docs


if __name__ == "__main__":
    docs = load_docs()
    for d in docs:
        print(f"- {d['source']}: {len(d['text'])} 字符")
