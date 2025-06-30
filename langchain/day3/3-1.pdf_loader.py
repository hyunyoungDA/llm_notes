from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader # 속도 최적화가 되어 있는 로더, 페이지 당 하나의 문서 반환 

FILE_PATH = "../data/SPRI_AI_Brief_2023년12월호_F.pdf"

# metadata 확인 함수 
def show_metadata(docs):
  if docs:
    print("[metadata]")
    print(list(docs[0].metadata.keys()))
    print("\n[examples]")
    max_key_length = max(len(k) for k in docs[0].metadata.keys())
    
    # 출력 시 ":" 동일한 위치에 설정 
    for k,v in docs[0].metadata.items():
      print(f"{k:<{max_key_length}} : {v}")

# 텍스트로 구성되어있는 PDF
loader1 = PyPDFLoader(FILE_PATH)

docs = loader1.load()

print(docs[10].page_content[:300])

show_metadata(docs)

# 페이지당 하나의 문서로 반환  
loader2 = PyMuPDFLoader(FILE_PATH)
pages = loader2.load()

print(docs[10].page_content[:300])

## pip install rapidocr-onnxruntime를 설치하면, PDF 내에 존재하는 이미지에서 텍스트를 추출할 수 있음 
loader_ocr = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)

ocr_pages = loader_ocr.load()
print(docs[4].page_content[:300])