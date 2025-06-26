from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
)

# {country}의 수도를 물어보는 체인 생성 
chain1 = (
    PromptTemplate.from_template("{country} 의 수도는 어디야?")
    | model
    | StrOutputParser()
)

# {country} 의 면적을 물어보는 체인을 생성
chain2 = (
    PromptTemplate.from_template("{country} 의 면적은 얼마야?")
    | model
    | StrOutputParser()
)

# 위의 2개 체인을 동시에 생성하는 병렬 실행 체인을 결합 
combined = RunnableParallel(capital=chain1, area=chain2)

result1 = chain1.invoke({"country":'대한민국'})
result2 = chain2.invoke({'country':'미국'})

print(f"result1: {result1}")
print(f"result2: {result2}")

# RunnableParallel 객체의 invoke 메서드는 주어진 country에 대한 처리 수행 
# 모든 체인에 대해 country의 값을 동일한 값으로 대입 
combined_result = combined.invoke({'country':'대한민국'})
print(f"combined_result: {combined_result}")
