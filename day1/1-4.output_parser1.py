# LLM을 이용해 가장 일반적으로 생성하는 구조화된 형식은 JSON

from langchain_ollama import ChatOllama
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
  '''사용자의 질문에 대한 답변과 그에 대한 근거(justification)를 함꼐 제공하세요.'''
  answer: str
  '''사용자의 질문에 대한 답변'''
  justification: str # LLM이 이해하기 쉽게 description 줌 
  '''답변에 대한 근거'''
  
llm = ChatOllama(
  model = 'gemma3:latest',
  base_url="http://localhost:11434",
  temperature=0.7,
)

# 스키마를 JSONSchema객체로 변환하여 LLM에 전송 -> 이를 통해 LLM에게 answer와 justification이라는 두 가지
# 필드를 가진 JSON 객체 형태로 응답을 줘야한다고 지시를 내리는 것
structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke('''1 킬로그램의 벽돌과 1킬로그램의 깃털 중 어느 쪽이 더 무겁나요?''')

# Pydantic 객체를 json 문자열 형태로 출력 
print(result.model_dump_json())