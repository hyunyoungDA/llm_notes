from langchain_community.embeddings import OllamaEmbeddings
import numpy as np

ollama_embeddings = OllamaEmbeddings(
  model = "chatfire/bge-m3:q8_0"
)

texts = [
    "안녕, 만나서 반가워.",
    "LangChain simplifies the process of building applications with large language models",
    "랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. ",
    "LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]

# 쿼리 임베딩
embedded_query = ollama_embeddings.embed_query("LangChain에 대해서 상세히 알려주세요")
len(embedded_query)

# 문서 임베딩
embedded_docs = ollama_embeddings.embed_documents(texts)

# 유사도 계산
similarity = np.matmul(np.array(embedded_query), np.array(embedded_docs).T)
# 유사도를 기준으로 내림차순 정렬
sorted_idx = (np.array(embedded_query) @ np.array(embedded_docs).T).argsort()[::-1]

for i, idx in enumerate(sorted_idx):
  print(f"[{i}] 유사도: {similarity[idx]:.3f} | {texts[idx]}")
  print()