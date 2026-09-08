"""文档加载：把知识库文件读成 Python 中的字典列表"""

from pathlib import Path

# 使用代码文件的位置定位资料，可以避免“在 src 下运行却找不到文档”的问题。
DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"


def load_docs(data_dir: str | Path = DOCS_DIR) -> list[dict]:
    """返回示例：
    [
        {
            "source": "校园生活.txt",
            "text": "学校的宿舍……"
        }
    ]
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"文档目录不存在：{data_dir}")
    docs = []
    print(f"[INFO] 开始加载 {data_dir} 下的所有文档")
    # glob 只找当前目录下的 .txt 文件；sorted 固定顺序，方便重复入库和排查。
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        # splitlines() 按行拆开，strip() 去掉每行首尾空白，空行不放入新列表。
        # join() 再把保留的行用换行符拼起来，得到清理后的全文。
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        cleaned = "\n".join(lines)
        docs.append({"source": path.name, "text": cleaned})
    print(f"[OK] 已加载 {len(docs)} 份文档")
    return docs

def main():
    # 可单独运行 python data_loader.py，先确认文件能读到
    docs = load_docs()
    for d in docs:
        print(f"- {d['source']}: {len(d['text'])} 字符")


if __name__ == "__main__":
    main()
