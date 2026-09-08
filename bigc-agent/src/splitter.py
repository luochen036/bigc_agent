"""文本切块：让检索针对具体段落，而不是整篇长文"""
import re


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """输入全文字符串，输出文本块列表。

    chunk_size 是每块的目标字符数，overlap 是相邻块重叠的字符数。
    这里按完整句子合并，所以单个长句可能让文本块超过目标长度。
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size 必须大于 0，overlap 必须在 0 到 chunk_size - 1 之间")

    sentences = re.split(r"(?<=[。！？；\n])", text)
    chunks = []
    current = ""
    for sentence in sentences:
        # 去掉句子首尾的空格，避免空字符串进入列表
        sentence = sentence.strip()
        if not sentence:
            continue

        if current and len(current) + len(sentence) > chunk_size:
            # 新句子放不下时，先把当前块存好，再开始下一块。
            chunks.append(current)
            # overlap 为 0 时要单独处理，因为字符串[-0:] 实际上会取整个字符串。
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + sentence
        else:
            current += sentence

    # 循环结束后，最后一个尚未存入列表的文本块也不能遗漏。
    if current:
        chunks.append(current)
    return chunks

def main():
    # 使用较小的目标长度演示切块。正式入库仍使用默认的 300 和 50。
    text = "北京印刷学院创办于1958年。\n学校位于北京市大兴区。\n校训是守正出新、笃志敏行！"
    for i, c in enumerate(split_text(text, chunk_size=30, overlap=5)):
        print(f"[{i}] {len(c)}字: {c}")



if __name__ == "__main__":
    main()
