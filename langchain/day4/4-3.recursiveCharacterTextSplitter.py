# 단락 -> 문장 -> 단어 순으로 재귀적 분할
# 이는 단락 단위가 의미적으로 가장 강하게 연관된 텍스트 조각으로 간주되기 때문

from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "../data/test.txt"

with open(FILE_PATH, encoding = 'utf-8') as f:
  file = f.read()
  
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 250,
  chunk_overlap = 50,
  length_function = len,
  is_separator_regex=False,
)

texts = text_splitter.create_documents([file])

print(texts[0])
print("===" * 20)

print(texts[1])

# file 텍스트 분할; 텍스트를 분할하고 분할된 텍스트의 처음 2개 요소 반환 
result = text_splitter.split_text(file)[:2]
print(result)