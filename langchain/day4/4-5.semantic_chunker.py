"""
텍스트를 의미론적 유사성에 기반하여 분할하며, 텍스트를 문장 단위로 분할한 후 3개의 문장씩 그룹화하고
임베딩 공간에서 유사한 문장들을 병합하는 과정을 거침 
"""
# from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

FILE_PATH = "../data/test.txt"

embeddings = OllamaEmbeddings(model = "chatfire/bge-m3:q8_0")

with open(FILE_PATH, encoding = 'utf-8') as f:
  file = f.read()
 
# 임베딩을 통해 의미론적 청크 분할기를 초기화 
# 보틍은 splitter로 chunk를 만든 후 embedding을 하지만 SemanticChunker는 역순으로 진행한 후 그룹화 
text_splitter = SemanticChunker(embeddings)

texts = text_splitter.split_text(file)
print(texts[0])
  