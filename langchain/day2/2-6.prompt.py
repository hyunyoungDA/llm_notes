from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
  model = "gemma3:latest",
  base_url="http://localhost:11434",
  temperature=0.7,
  top_p = 0.9,
)

# 템플릿 생성과 동시에 prompt 생성

template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    # 사용자가 채워야 하는 값
    input_variables=["country1"],
    # 부분 변수 채움 -> 항상 공통된 방식으로 가져오고 싶은 변수가 있는 경우(날씨, 현재 시간)
    # 항상 정의한 값으로 고정되어 들어감 
    partial_variables = {
      "country2": "미국" # dictionary 형태로 partial_variables 전달 
    }
)

# format()은 partial_variables와 같이 조합된 템플릿을 처리하지 못함
# prompt.format_prompt(country1 = "대한민국").to_string() # 텍스트 문자 그대로 반환 
print(prompt.format(country1="대한민국")) # 현재는 input_variables만 채우면 되므로, format으로도 처리 가능 

chain = prompt | model
result = chain.invoke({"country1": "대한민국"})

print(result)