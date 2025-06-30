from langchain_text_splitters import CharacterTextSplitter

FILE_PATH = "../data/test.txt"

with open(FILE_PATH, encoding = 'utf-8') as f:
  file = f.read()
  
  # print(file)

text_splitter = CharacterTextSplitter(
  #separator="" # 기본 값은 "\n\n"
  chunk_size = 250,
  chunk_overlap = 50,
  length_function = len, # 텍스트 길이 계산 함수 지정 
  is_separator_regex=False # separator를 정규식이 아닌 일반 문자열로 처리 
)

texts = text_splitter.create_documents([file])
print(texts[0]) # 분할된 문서 중 첫 번째 문서 출력 
