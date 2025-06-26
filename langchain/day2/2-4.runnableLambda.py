from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


llm = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
  top_p = 0.9,
)

def get_today(a):
  return datetime.today().strftime("%b-%d")

prompt = PromptTemplate.from_template(
  "{today}가 생일인 유명한 {n}명을 나열하세요. 생년월일을 표기해주세요"
)

# RunnableLambda는 사용자 정의 함수 맵핑 가능 
chain = (
  {"today" : RunnableLambda(get_today), "n":RunnablePassthrough()
   | prompt
   | llm
   | StrOutputParser()}
)