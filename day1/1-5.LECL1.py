from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


template = '''
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어
회화를 작성해주세요.

상황: {question}

FORMAT:
- 영어 회화:
- 한글 해석:
'''

prompt = PromptTemplate.from_template(template)

model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# stream은 모든 응답이 완료된 후 처리하는 invoke와는 다르게 첫 글자를 생성하는 순간부터 그 글자를 즉시 받음 
# 응답의 청크를 스트리밍하여 메모리 효율이 좋음 
for chunk in chain.stream({'question':"저는 식당에 가서 음식을 주문하고 싶어요"}):
  print(chunk, end = "", flush = True) # flush = True는 버퍼링 없이 출력

print("\n")
