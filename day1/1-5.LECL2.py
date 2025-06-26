from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
)

prompt = PromptTemplate.from_template("{topic}에 대해 3문장으로 설명해줘")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

#batch: 배치 단위 실행; 여러 개의 딕셔너리를 포함하는 리스트를 인자를 받아서 일괄 처리
# max_concurrency를 통해 동시에 처리할 수 있는 최대 작업 수 설정 
response = chain.batch([
  {'topic':'ChatGPT'},
  {'topic':"Instagram"},
  {'topic':"멀티모달,"}
  ],
  config = {'max_concurrency':2})
print(response)