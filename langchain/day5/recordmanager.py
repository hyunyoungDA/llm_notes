# 벡터 저장소는 데이터가 변화할때마다 데이터를 다시 인덱싱
# 이는 계산 비용이 발생하며 내용 중복

from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from sqlalchemy import create_engine

CONNECTION_STRING = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
COLLECTION_NAME = "my_rag_documents_from_test_txt_final" # 새로운 컬렉션 이름으로 구분
NAME_SPACE = 'my_docs_namespace'

embeddings_model = OllamaEmbeddings(
    model = 'nomic-embed-text',
    base_url = "http://localhost:11434",
)

# PGVector가 connection_string 대신 SQLAlchemy Engine을 기대합니다.
db_engine = create_engine(CONNECTION_STRING)

record_manager = SQLRecordManager(
    namespace = NAME_SPACE,
    db_url = CONNECTION_STRING
)

record_manager.create_schema()

docs = [
    Document(page_content = 'there are cats in the pond', metadata = {
        'id':1, 'source':'cat.txt'
    }),
    Document(page_content = "ducks are also found in the pond", metadata = {
        'id':2,'source':'ducks.txt'
    })
]

# postgres.vectorstore 에서는 PGVector에서 바로 받음 
vectorstore = PGVector(
        embeddings=embeddings_model,
        #documents=docs,
        collection_name=COLLECTION_NAME,
        connection=db_engine
    )

# 여기서 PGVector에 docs를 전달함
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental', # 문서 중복 장지
    source_id_key = 'source' # 출처를 source_id로 사용 
)

print(f"인덱싱 1회차:{index_1}")

# 문서 인덱싱 2회차, 중복 문서 생성 X
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup='incremental', # 문서 중복 장지
    source_id_key = 'source' # 출처를 source_id로 사용 
)
print(f"인덱싱 2회차:{index_2}")

# 문서를 수정하면 새 버전을 저장, 출처가 같은 기존 문서는 삭제
docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup = "incremental",
    source_id_key = 'source'
)