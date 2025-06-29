import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

FILE_PATH = "../data/titanic.csv"

df = pd.read_csv(FILE_PATH)
# df = pd.read_excel("./data/titanic.xlsx") 처럼 엑셀도 df로 불러온 후 처리 가능 

loader = DataFrameLoader(df, page_content_column="Name")
# print(df.head())

docs = loader.load()

print(docs[0].page_content)

print(docs[0].metadata) # 메타 데이터 출력 