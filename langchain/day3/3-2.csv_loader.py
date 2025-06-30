from langchain_community.document_loaders.csv_loader import CSVLoader

FILE_PATH = "../data/titanic.csv"

loader = CSVLoader(FILE_PATH)

docs = loader.load()

print(len(docs))
print(docs[0].metadata)
print("=" * 30)

## CSV Parsing

loader_parse = CSVLoader(
  file_path = FILE_PATH,
  # source_column: 각 행에서 생성된 문서의 출처 지정 
  csv_args = {
    "delimiter":"," ,# 구분자
    "quotechar": '"',# 인용 부호 문자
    "fieldnames": [
            "Passenger ID",
            "Survival (1: Survived, 0: Died)",
            "Passenger Class",
            "Name",
            "Sex",
            "Age",
            "Number of Siblings/Spouses Aboard",
            "Number of Parents/Children Aboard",
            "Ticket Number",
            "Fare",
            "Cabin",
            "Port of Embarkation",
        ],  # 필드 이름
  }
)

docs1 = loader_parse.load()

print(docs1[1].page_content)
print("=" * 30)

loader_column = CSVLoader(
  file_path = FILE_PATH, source_column = "PassengerId"
  # source_column을 통해 각 행에서 생성된 문서의 출처 지정
  # 여기서는 PassengerId로 출처 지정 
)

docs2 = loader_column.load()

print(docs2[1])