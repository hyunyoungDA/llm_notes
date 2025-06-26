from operator import itemgetter # 특정 키 추출 라이브러리

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# 길이 반환 함수
def length_function(text):
  return len(text)

def _multiple_length_function(text1, text2):
  return len(text1) * len(text2)

# 두 문장의 길이를 곱한 값을 반환하는 함수 
def multiple_length_function(_dict):
  return _multiple_length_function(_dict["text1"], _dict['text2'])

prompt = ChatPromptTemplate.from_template(
  "{a} + {b}는 무엇인가요?"
)

model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
  top_p = 0.9,
)

chain1 = prompt | model

chain = (
    { 
        # a는 바로 length_function에 대입 
        # itemgetter는 word1 키에 대한 값을 추출 
        "a": itemgetter("word1") | RunnableLambda(length_function),
        # 
        "b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
)

result = chain.invoke({
  "word1":"apple",
  "word2":"banana",
})

print(result)