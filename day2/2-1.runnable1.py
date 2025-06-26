from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요")

model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
)

# 출력 파서 
output_parser = StrOutputParser()

# 다양한 구성 요소를 단일 체인으로 결합 가능 
chain = prompt | model | output_parser

input_ = {'topic':"바다코끼리"}

answer = chain.invoke(input_)

print(answer)