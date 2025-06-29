# 간단한 문자 텍스트 분할 
from langchain_text_splitters import CharacterTextSplitter

FILE_PATH = "../data/test.txt"

with open(FILE_PATH, encoding = 'utf-8') as f:
  file = f.read()

# 문서에 대한 메타데이터 리스트 정의 
metadatas = [
  {'document': 1},
  {"document": 2},
]

text_splitter = CharacterTextSplitter(
  chunk_size= 250,
  chunk_overlap = 50,
  length_function = len,
  is_separator_regex=False,
)

documents = text_splitter.create_documents(
  [
    file,
    file,
  ], # 분할할 텍스트 데이터를 리스트로 전달
  metadatas = metadatas
)

print(documents[1])