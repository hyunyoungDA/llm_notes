# UNSET SSL_CERT_FILE
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS # FAISS 인덱싱 
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

model = ChatOllama(
  model = "gemma3:latest",
  base_url = "http://localhost:11434",
  temperature=0.7,
)

# txt 문서 불러오기
loader = TextLoader("../data/test.txt")

documents = loader.load() # 문서 로드

text_splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
)

# 로드된 분서를 splitter로 분할 
split_docs = text_splitter.split_documents(documents)

# ollama 임베딩 모델 로드
ollama_embeddings = OllamaEmbeddings(
  model = "chatfire/bge-m3:q8_0"
)
# 분할된 텍스트와 임베딩 사용하여 FAISS 벡터 DB 생성 
db = FAISS.from_documents(split_docs, ollama_embeddings)

# 데이터베이스를 검색기로 사용하기 위해 retriever 변수에 할당 
"""
search_type: 검색 유형("similarity","mmr","similarity_score_threshold")
search_kwags: 추가 검색 옵션
    k: 반활할 문서 수 (default = 4)
    score_threshold: similarity_score_threshold 검색의 최소 유사도 임곗값
    fetch_k: MMR 알고리즘에 전달할 문서 수 (default = 20)
    lambda_mult: MMR 결과의 다양성 조절(0-1 사이, default = 0.5); 1에 가까울수록 유사성이 높음 
    
반환값 -> VectorStoreRetriever 객체 
"""
retriever = db.as_retriever()
"""
1. MMR 방식은 쿼리에 대한 관련 항목을 검색할 때 검색된 문서의 중복을 피하는 방법 중 하나임
단순히 가장 관련성 높은 항목들만 검색하는 것이 아닌, 쿼리와 문서의 관련성과 이미 선택된 문서들과의 차별성 모두 고려 
retriever = db.as_retriever(
    search_type = "mmr", search_kwargs = {"k":2, "fetch_k": 10,"lambda_mult":0.6}
)

2. 유사도 점수 임계값 검색
retriever = db.as_retriever(
    # 검색 유형을 "similarity_score_threshold 으로 설정
    search_type="similarity_score_threshold",
    # 임계값 설정
    search_kwargs={"score_threshold": 0.8},
)
"""


# retriever에서 invoke는 관련 문서 검색하는데 사용 
docs = retriever.invoke("임베딩은 무엇인가요?")

for doc in docs:
    print(doc.page_content)
    print("====================")