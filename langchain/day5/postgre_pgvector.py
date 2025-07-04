import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PostgreSQL 연결 정보
# 'postgresql+psycopg://사용자이름:비밀번호@호스트:포트/데이터베이스이름'
CONNECTION_STRING = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
COLLECTION_NAME = "my_rag_documents_from_test_txt_final" # 새로운 컬렉션 이름으로 구분

# embeddings 모델 정의 
ollama_base_url = "http://localhost:11434"
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)

# 절대 경로 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(current_script_dir, "data", "test.txt")

# 파일 존재 여부 다시 확인 
if not os.path.exists(FILE_PATH):
    print(f"오류: 지정된 파일 '{FILE_PATH}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
    exit() # 파일이 없으면 스크립트 종료

print(f"'{FILE_PATH}' 파일에서 문서를 로드합니다.")
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

# 문서 로드후 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

print(f"총 {len(documents)} document(s)를 로드했고, {len(texts)} chunk(s)로 분할했습니다.")

# PGVector에 임베딩 저장
print("PGVector에 임베딩 저장 중...")
try:
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print("임베딩이 PGVector에 성공적으로 저장되었습니다!")

    # 저장된 임베딩 검색 (확인용)
    print("\n유사성 검색을 통해 저장 확인 중...")
    # query를 db로 임베딩하여 가장 유사한 임베딩 1개 검색 
    query = "LangChain의 주요 기능은 무엇인가요?"
    docs_with_score = db.similarity_search_with_score(query, k=1)
    if docs_with_score:
        for doc, score in docs_with_score:
            print(f"검색된 문서: {doc.page_content}")
            print(f"유사성 점수: {score}")
    else:
        print("검색 결과가 없습니다.")

except Exception as e:
    print(f"오류 발생: {e}")
    print("다음 사항을 다시 확인해주세요:")
    print(f"- PostgreSQL 컨테이너 ({os.environ.get('POSTGRES_HOST')}:{os.environ.get('POSTGRES_PORT')})가 실행 중인지")
    print(f"- 연결 문자열 '{CONNECTION_STRING}'이 정확한지")
    print(f"- Ollama 서버가 '{ollama_base_url}'에서 실행 중인지")
    print("- 'nomic-embed-text' 모델이 Ollama에 다운로드되어 있는지 (`ollama list` 확인)")
