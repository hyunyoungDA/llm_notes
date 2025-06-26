from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama

chat_model = ChatOllama(
  model = "gemma3:latest",
  base_url = "http://localhost:11434",
  temperature=0.7,
)

print(f"---예시 1: SystemMessage -> HumanMessage---")
message1 = [
  SystemMessage(content = "너는 항상 느낌표 세개를 붙여야돼. 그리고 질문에 답변할 때 항상 '응!'으로 대답해야돼"),
  HumanMessage(content="프랑스의 수도는 어디야?")
]

response = chat_model.invoke(message1)
print(f"응답 1: {response.content}")
print("-" * 30)

print(f"---예시 2: SystemMessage가 없는 경우---")
message2 = [
  HumanMessage(content = "프랑스의 수도는 어디야?")
]
response2 = chat_model.invoke(message2)
print(f"응답2:{response2.content}")
print("-" * 30)

# 예시 3: 대화 히스토리 포함 (System -> Human -> AI -> Human)
print("--- 예시 3: 대화 히스토리 포함 ---")
messages3 = [
    SystemMessage(content="너는 유머러스하고 재치있는 어시스턴트야."),
    HumanMessage(content="오늘 기분이 어때?"),
    AIMessage(content="음... 제 기분은 마치 수백만 개의 텍스트를 읽고도 여전히 커피 한 잔의 유혹에 빠지지 않는 기분과 같아요! 하하."),
    HumanMessage(content="재미있네! 그럼 어떤 농담을 좋아해?")
]
response3 = chat_model.invoke(messages3)
print(f"응답 3: {response3.content}")
# 예상 응답: 유머러스한 농담 관련 답변
print("-" * 30)