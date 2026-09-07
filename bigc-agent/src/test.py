from store import get_collection
from embedder import load_model
from retriever import retrieve

collection = get_collection()
model = load_model()

results = retrieve(
      collection=collection,
      query="北京印刷学院有哪些特色专业？",
      embedder=model,
      k=3,
      min_score=0.0,
  )

for item in results:
    print("来源：", item["source"])
    print("相似度：", round(item["score"], 3))
    print("内容：", item["text"])
    print("="*25)