# 北京印刷学院文档问答：入门教学版

这个项目演示最基础的 RAG（检索增强生成）：先找资料，再让大模型根据资料回答。
目标是看懂一条完整的数据处理流程。当前只提供命令行单次问答，没有网页、对话记忆或工具调用。

## 一、先运行起来

你已经配置好的 `bigc-agent/.env` 会继续使用，不需要重新填写。

如果终端已经激活虚拟环境，并且当前目录是 `bigc-agent/src`，只需执行：

```powershell
python build_index.py
python agent.py
```

第二条命令会显示“请输入问题：”，输入“北京印刷学院的校训是什么？”并回车。
程序回答一次后结束，想问下一个问题就再次运行 `python agent.py`。

第一条命令是准备资料，只在首次使用或文档改变后运行；不是每次提问前都要运行。

如果从仓库根目录开始，PowerShell 中的完整步骤是：

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd bigc-agent\src
python build_index.py
python agent.py
```

CMD 的虚拟环境激活命令是 `venv\Scripts\activate.bat`。
若根目录还没有 venv，先用 `python -m venv venv` 创建。使用 Python 3.10 或更新版本，
具体依赖是否支持你的 Python 版本，以安装结果为准。

## 二、先分清两种模型

| 模型 | 输入 | 输出 | 在哪里运行 |
| --- | --- | --- | --- |
| 嵌入模型 | 一段文档或一个问题 | 一组数字，即向量 | 本地电脑 |
| 聊天模型 | 系统规则、参考资料、问题 | 自然语言回答 | .env 配置的服务 |

向量可以帮助比较文字含义。例如“学校有哪些专业”和“介绍一下学校专业设置”，
字面不同，但含义接近，模型生成的向量通常也比较接近。

本项目使用现成的模型，不训练模型。把文档写入 Chroma 是建立检索索引，
不等于把文档知识训练进聊天模型。

## 三、先看整体流程

准备资料时：

```text
.txt 文档
  → 读取并清理文本
  → 切成多个文本块
  → 本地嵌入模型生成向量
  → Chroma 保存原文、来源、向量
```

回答问题时：

```text
用户输入问题
  → 同一个嵌入模型生成问题向量
  → Chroma 找出相似文本块
  → 将文本块和问题组成提示词
  → 聊天模型生成回答
  → 显示回答和参考来源
```

没有文本块通过相似度阈值时，程序直接回复“现有资料中没有相关信息”，不调用聊天模型。

## 四、按顺序阅读代码

代码中已有中文模块说明、函数说明和关键语句注释。建议按以下顺序阅读，
每一步先关注“输入是什么、输出是什么”，再看具体语法。

### 1. data_loader.py：把文件读进 Python

文件位置：[data_loader.py](bigc-agent/src/data_loader.py)。

输入是 `data/docs/` 目录中的 UTF-8 编码 `.txt` 文件，只读取直接子文件。
输出是一个列表，列表里每个字典代表一篇文档：

```python
[
    {"source": "校园生活.txt", "text": "清理后的全文"},
    {"source": "特色专业.txt", "text": "清理后的全文"},
]
```

列表保存多篇文档，字典保存一篇文档的不同属性。
`doc["text"]` 取正文，`doc["source"]` 取文件名。

路径从代码文件的位置计算，因此在 `src` 或项目根目录启动都能找到同一份资料。
可以在 `src` 中单独运行 `python data_loader.py`，先观察加载结果。

### 2. splitter.py：把长文切小

文件位置：[splitter.py](bigc-agent/src/splitter.py)。

输入是一篇完整文章，输出是 `["文本块1", "文本块2", ...]`。
每块目标约 300 字，相邻块保留约 50 字重复内容，帮助保留边界上下文。
按句子合并时，遇到很长的句子可能超过 300 字，这不是严格长度上限。

这样检索可以定位到具体段落，也避免把全部文档都发送给聊天模型。
单独运行 `python splitter.py` 可以查看小样例。

### 3. embedder.py：把文字变为向量

文件位置：[embedder.py](bigc-agent/src/embedder.py)。

`load_model()` 加载本地的 `BAAI/bge-small-zh-v1.5` 模型，返回模型对象。
本地模型目录不存在时，库会尝试联网下载。

```python
model = load_model()
vectors = model.encode(["文本块1", "文本块2"], normalize_embeddings=True)
```

一段文本对应一个向量；两段文本就得到两个向量。
`normalize_embeddings=True` 将向量长度统一为 1，便于比较方向相似度。
文档和问题必须使用同一个嵌入模型，向量才有可比性。

单独运行 `python embedder.py` 会显示一个问题的向量形状和前几个数字。

### 4. store.py：将资料存到数据库

文件位置：[store.py](bigc-agent/src/store.py)。

Chroma 集合可以先理解为一张表，每条记录是一个文本块：

| 字段 | 保存什么 | 为什么需要 |
| --- | --- | --- |
| ids | 唯一编号 | 重复入库时识别同一位置的记录 |
| documents | 文本块原文 | 检索后交给聊天模型阅读 |
| embeddings | 文本向量 | 比较问题和资料是否相似 |
| metadatas | 来源文件名 | 回答时能追溯出处 |

`get_collection()` 打开数据库，`index_docs()` 写入当前全部文档的块。
三个输入列表按位置对应：第一个向量来自第一个文本块，它的出处是第一个来源。

数据保存在 `bigc-agent/storage/chroma/`，关闭程序也不会丢失。
每次重新入库都会更新内容并清理多余旧块；空文档集合会报错并保留原索引。
这里按整套文档更新，不支持只传一篇文档做增量导入。更新时先停止问答程序。

### 5. build_index.py：串起准备阶段

文件位置：[build_index.py](bigc-agent/src/build_index.py)。

这个文件依次调用刚才的四个模块：

```text
load_docs → split_text → model.encode → index_docs
```

重点看 `chunks`、`sources`、`embeddings` 三个列表如何一一对应。
主入口只加载一次嵌入模型，再把所有文本块交给它处理。

### 6. retriever.py：找出相关段落

文件位置：[retriever.py](bigc-agent/src/retriever.py)。

输入：数据库集合、问题、嵌入模型。
输出：包含原文、来源和相似度的字典列表。

```python
[
    {"text": "命中的资料原文", "source": "特色专业.txt", "score": 0.82},
]
```

默认最多取 3 个块，过滤掉相似度低于 0.3 的结果。
Chroma 被配置为余弦距离，代码通过 `score = 1 - distance` 得到余弦相似度。
分数越大通常越相关，但不是“答案正确率”，0.3 也不是经过评测的通用阈值。

在 `src` 运行 `python test.py`，可以只观察检索原文，不调用聊天模型。
检索效果不好时先检查这一步，再检查原文，不要只盯着最终回答。

### 7. languagemodel.py：调用聊天服务

文件位置：[languagemodel.py](bigc-agent/src/languagemodel.py)。

`load_dotenv()` 加载配置，`os.getenv()` 读取配置：
`LLM_BASE_URL` 决定请求地址，`LLM_API_KEY` 用于认证，`LLM_MODEL` 指定模型。
已有的系统环境变量优先于 `.env`，实际配置文件被 Git 忽略。

`generate_answer()` 把两条消息发送给兼容接口：
`system` 是系统规则，`user` 是“参考资料 + 问题”。
接口返回对象，`response.choices[0].message.content` 才是回答文字。

这一步会联网，将问题和命中的文本块发送给配置的模型服务，并可能产生接口费用。
导入这个模块不会请求接口，只有调用函数才会请求。

### 8. agent.py：串起问答阶段

文件位置：[agent.py](bigc-agent/src/agent.py)。

先看 `answer_question()`，它只做三件事：

1. 调用 `retrieve()` 找资料。
2. 调用 `build_prompt()` 给资料编号，再拼上问题。
3. 调用 `generate_answer()` 得到回答。

再看 `main()`：读取用户输入，打开数据库，加载嵌入模型，调用上述函数并打印结果。
`if __name__ == "__main__"` 让这个入口只在直接运行文件时执行，
被其他文件导入时不会突然询问用户或请求模型。

模型被要求使用 `[1]` 等来源编号，但提示词不能保证它绝不出错。
终端列出的来源是本次检索结果，阅读答案时仍应回看原文。

## 五、教学版的范围

现在只需记住两个正式入口：`build_index.py` 和 `agent.py`。
`test.py` 是可选的检索练习；`tests/test_pipeline.py` 是供维护使用的自动化测试，
初次学习可以跳过。未使用的网页依赖、缓存装饰器和空评测文件已移除。

旧版的命令行问题参数、`--dry-run`、`--top-k`、`--min-score`、`--db-path`、
`--data-dir` 和连续提问模式已经移除。直接运行 `store.py` 也不再执行入库。
学习时需要调整检索参数，可以修改 `retriever.py` 中函数的默认值。

验证核心功能的命令，在 `bigc-agent` 目录执行：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

这些测试使用模拟聊天响应，不需要真实密钥，也不会调用远程聊天接口。
测试通过代表相应代码行为符合预期，不代表知识库事实或模型回答已被全面验证。
