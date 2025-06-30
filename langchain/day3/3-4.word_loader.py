from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

FILE_PATH = "../data/2 관련 연구부분(수정1).docx"
loader = Docx2txtLoader(FILE_PATH)

docs = loader.load()

print(len(docs))
print(docs)
print('=' * 30)

## 비정형 데이터 로드
# 내부적으로 비정형은 텍스트 덩어리마다 서로 다른 요소를 만드는데, mode = "elements"를 지정하여 쉽게 분리 가능 
unstructured_loader = UnstructuredWordDocumentLoader(
  FILE_PATH, mode = "elements" )

docs2 = unstructured_loader.load()

print(docs2)