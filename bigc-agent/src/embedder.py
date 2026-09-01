from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 轻量中文嵌入模型，512 维


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    vec = model.encode("北京印刷学院创办于哪一年？")
    print(vec.shape)  # (512,) —— 一句话 → 512 维向量
    print(vec[:5])  # 看一眼向量长什么样


if __name__ == "__main__":
    main()
