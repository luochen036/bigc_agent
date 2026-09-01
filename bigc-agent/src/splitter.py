import re


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """滑动窗口切分：按句子切 → 按窗口合并 → 保留 overlap 上下文。"""
    sentences = re.split(r"(?<=[。！？；\n])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > chunk_size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + s          # 用上一块的结尾衔接上下文
        else:
            cur += s
    if cur:
        chunks.append(cur)
    return chunks


if __name__ == "__main__":
    text = "北京印刷学院创办于1958年。\n学校位于北京市大兴区。\n校训是守正出新、笃志敏行。"
    for i, c in enumerate(split_text(text)):
        print(f"[{i}] {len(c)}字: {c}")
