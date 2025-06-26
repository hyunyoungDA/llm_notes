## CSV 처리
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

items = parser.invoke("apple, nanana, cherry")
print(items)