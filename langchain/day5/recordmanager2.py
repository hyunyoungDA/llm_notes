from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL # URL 임포트 추가

# 1. 연결 및 설정 정보
# connection_string을 그대로 사용하되, create_engine에서 URL 객체를 활용
DB_USER = 'langchain'
DB_PASSWORD = 'langchain'
DB_HOST = '127.0.0.1'
DB_PORT = 6024
DB_NAME = 'langchain'

# SQLAlchemy URL 객체 생성 (drivername 명시!)
# 이 부분이 가장 중요합니다. psycopg 드라이버가 vector 타입을 인식하도록 돕습니다.
db_url_object = URL.create(
    drivername="postgresql+psycopg", # 명시적으로 psycopg 드라이버 사용
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# create_engine에 URL 객체 전달
db_engine = create_engine(db_url_object)

COLLECTION_NAME = "my_rag_documents_from_test_txt_final"
NAME_SPACE = 'my_docs_namespace'

# 2. 임베딩 모델 초기화
embeddings_model = OllamaEmbeddings(
    model = 'nomic-embed-text',
    base_url = "http://localhost:11434",
)

# 3. SQLRecordManager 초기화 및 스키마 생성
record_manager = SQLRecordManager(
    namespace = NAME_SPACE,
    # db_url은 여전히 문자열 형태가 편리합니다.
    # SQLRecordManager는 URL 객체를 바로 받지 않을 수 있으므로, 기존 문자열 사용.
    db_url = str(db_url_object) # URL 객체를 문자열로 변환하여 전달
)
record_manager.create_schema()

# 4. 문서 정의
docs = [
    Document(page_content = 'there are cats in the pond', metadata = {
        'id':1, 'source':'cat.txt'
    }),
    Document(page_content = "ducks are also found in the pond", metadata = {
        'id':2,'source':'ducks.txt'
    })
]

# 5. PGVector 인스턴스 생성
# connection 매개변수에 SQLAlchemy Engine 객체 전달
vectorstore = PGVector(
    embedding=embeddings_model,
    collection_name=COLLECTION_NAME,
    connection=db_engine,
)

# --- 이하 인덱싱 코드는 동일 ---

print("인덱싱 1회차 시작...")
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental',
    source_id_key = 'source'
)
print(f"인덱싱 1회차:{index_1}")

print("\n인덱싱 2회차 시작...")
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental',
    source_id_key = 'source'
)
print(f"인덱싱 2회차:{index_2}")

print("\n문서 수정 후 인덱싱 3회차 시작...")
docs[0].page_content = "I just modified this document! This is a new version."

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup = "incremental",
    source_id_key = 'source'
)
print(f"인덱싱 3회차:{index_3}")