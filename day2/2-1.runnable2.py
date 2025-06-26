## RunnablePassthrough: 입력을 변경하지 않거나 추가 키를 더하여 전달할 수 있음 
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{num}의 10배는?")

llm = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature = 0.7,
)

chain = prompt | llm

# chain을 invoke할 때는 입력 데이터의 타입이 딕셔너리여야 함 
# 1개의 변수만 템플릿에 포함하고 있다면, 값만 전달 가능 
chain.invoke({"num": 5})

# 입력을 그대로 넘기고 싶을 때 유용 
runnable_chain = {'num': RunnablePassthrough()} | prompt | llm

# RunnablePassthrough.assign(): 입력 값으로 들어온 값의 key/value 쌍과 새롭게 할당된 key/value 쌍을 합침
result = (RunnablePassthrough.assign(new_num = lambda x: x['num'] * 3)).invoke({'num': 1})
print(f"assign 결과: {result}")