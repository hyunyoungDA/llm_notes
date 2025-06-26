from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature = 0.7,
)

# 여러 Runnable 인스턴스를 병렬로 처리 가능 
runnable = RunnableParallel(
  # 입력된 데이터를 그대로 통과 
  passed = RunnablePassthrough(),
  # mult 람다 함수를 통해서 딕셔너리의 num키에 해당하는 값의 *3을 저장 
  extra = RunnablePassthrough.assign(mult = lambda x: x['num'] * 3),
  # 입력된 딕셔너리의 "num"키에 해당하는 값 + 1
  modified = lambda x: x['num'] + 1,
)

result = runnable.invoke({'num': 1})
print(result)