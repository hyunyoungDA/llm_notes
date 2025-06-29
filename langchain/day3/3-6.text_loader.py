from langchain_community.document_loaders import TextLoader, DirectoryLoader

FILE_PATH = "../data/test.txt"

loader = TextLoader(FILE_PATH, encoding = "utf-8")

# 문서 로드
docs = loader.load()
print(f"문서의 수: {len(docs)}\n")
print("[메타데이터]\n")
print(docs[0].metadata)
print("\n========= [앞부분] 미리보기 =========\n")
print(docs[0].page_content[:500])

path = "data/"

# autodetect_encoding: 로더 클래스에 자동 감지_인코딩을 전달해서 실패하기 전에 파일 인코딩을 자동 감지 
text_loader_kwargs = {'autodetect_encoding': True}

loader1 = DirectoryLoader(
  path,
  glob = "**/*.txt", # .txt 파일만 
  loader_cls = TextLoader, # 각 파일을 TextLoader로 로드 
  slient_errors = True, # 디렉토리로더에서 silent_errors 매개변수를 전달하여 로드할 수 없는 파일을 건너뛰고 계속 진행 
  loader_kwargs = text_loader_kwargs,
)

docs = loader1.load()