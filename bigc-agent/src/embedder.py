"""嵌入模型：把文字转为向量，供数据库比较含义是否相近。"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 这是本项目已经下载好的模型目录。若没有本地文件，库会尝试在线下载。
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "BAAI--bge-small-zh-v1.5"
    / "snapshots"
    / "master"
)


def load_model() -> SentenceTransformer:
    """返回一个模型对象，之后用 model.encode([文本1, 文本2]) 得到向量。"""
    if LOCAL_MODEL_DIR.is_dir():
        model_source = str(LOCAL_MODEL_DIR)
    else:
        model_source = MODEL_NAME

    # 建索引和检索问题必须使用同一个嵌入模型，向量才处在同一个比较空间。
    return SentenceTransformer(model_source)


def main():
    # 测试模型是否加载成功。
    model = load_model()
    vectors = model.encode(["学校有哪些特色专业？"], normalize_embeddings=True)
    print("向量形状（文本数量，向量维度）：", vectors.shape)
    print("第一个向量的前 5 个数字：", vectors[0][:5])


if __name__ == "__main__":
    main()
