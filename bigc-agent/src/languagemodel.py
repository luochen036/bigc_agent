"""模型接口：读取 .env，把提示词发送给兼容 OpenAI 格式的聊天服务"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# __file__ 是当前代码文件；parents[1] 是它上两层的 bigc-agent 目录。
# 因此无论终端在哪个目录启动，读取的都是 bigc-agent/.env。
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_llm_config() -> dict[str, str]:
    """返回三个配置项组成的字典；缺少配置时先报错，不发网络请求。"""
    # 将 .env 的配置加载到环境变量中。已有的环境变量优先，不会被覆盖。
    load_dotenv(ENV_PATH, override=False)
    config = {}
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        # getenv 的第二个参数是默认值：没找到时返回空字符串。
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"请在 {ENV_PATH} 中填写 {name}")
        config[name] = value
    return config


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """输入系统规则和“资料 + 问题”，输出模型生成的回答文字。"""
    config = load_llm_config()
    with OpenAI(
        api_key=config["LLM_API_KEY"],
        base_url=config["LLM_BASE_URL"],
        timeout=60.0,
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=config["LLM_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # 较低的 temperature 通常减少随机性，但不等于保证回答正确。
            temperature=0.2,
        )

    if not response.choices:
        raise RuntimeError("模型服务没有返回文本回答")
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("模型服务没有返回文本回答")
    return answer.strip()
