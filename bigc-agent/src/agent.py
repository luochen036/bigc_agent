# src/agent.py —— Prompt 设计
SYSTEM_PROMPT = """你是「北京印刷学院智能体」，一名熟悉北京印刷学院的高校问答助手。

回答规则：
1. 只依据【参考资料】作答，禁止编造任何信息；
2. 若参考资料中没有答案，直接回答「现有资料中没有相关信息」；
3. 每条信息后标注来源编号，如 [1]；
4. 用简洁、有条理的书面中文回答。"""

def build_prompt(question: str, hits: list[dict]) -> str:
    refs = "\n\n".join(
        f"[{i+1}]（来源：{h['source']}）\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return f"【参考资料】\n{refs}\n\n【问题】\n{question}"