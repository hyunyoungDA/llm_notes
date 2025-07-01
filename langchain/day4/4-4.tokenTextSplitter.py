# 언어 모델은 토큰 제한이 있으므로, 토큰 제한을 초과하면 안 됨.
from langchain_text_splitters import CharacterTextSplitter
# 텍스트를 토큰 수를 기반으로 청크를 생성할 때 유용 
from langchain_text_splitters import TokenTextSplitter
from transformers import  AutoTokenizer 

FILE_PATH = "../data/test.txt"
with open(FILE_PATH, encoding = 'utf-8') as f:
  file = f.read()
  
# print(file[:500])

# 텍스트는 CharacterTextSplitter에 의해서만 분할되고, tiktoken 토크나이저는 분할된 텍스트를 병합하는 데 사용
# 분할된 텍스트가 tiktoken 토크나이저로 측정한 청크 크기보다 클 수 있음
# tiktoken은 OpenAI 모델의 토크나이저를 쓰기 때문에 OpenAI 사용시 유리, 토큰 길이를 기준으로 분할 
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  chunk_size = 300,
  chunk_overlap = 0,
)

texts = text_splitter.split_text(file)
print(len(texts))
print(texts[0])

# HuggingFace 또는 사용자 제공 토크나이저로 분할 
"""
tokenizer = AutoTokenizer("bert-base-uncased")
token_text_splitter = TokenTextSplitter(
  chunk_size = 200,
  chunk_overlap = 0,
  tokenizer = tokenzier
)
"""
token_text_splitter = TokenTextSplitter(
  chunk_size = 200,
  chunk_overlap = 0, 
)

texts = token_text_splitter.split_text(file)
print(texts[0])