"""可选练习：只观察检索结果，不调用远程聊天模型。

先运行 build_index.py，再运行 python test.py。
这与 tests/test_pipeline.py 的自动化测试不同，它供学习时手动查看结果。
"""

from store import get_collection
from embedder import load_model
from retriever import retrieve

if __name__ == "__main__":
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("索引为空，请先运行 build_index.py")
    model = load_model()

    # 可以修改这个问题，观察相同文档面对不同问题会得到什么检索结果。
    results = retrieve(collection, "北京印刷学院有哪些特色专业？", model)
    for item in results:
        print("来源：", item["source"])
        print("相似度：", round(item["score"], 3))
        print("内容：", item["text"])
        print("=" * 25)
    if not results:
        print("没有资料通过相似度阈值")
