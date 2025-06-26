from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


model = ChatOllama(
  model = 'gemma3:latest',
  base_url="http://localhost:11434",
  temperature=0.7
)

system_msg = SystemMessage(
  '''당신은 문장 끝에 느낌표를 세 개 붙여 대답하는 친절한 어시스턴트야.'''
)

human_msg = HumanMessage('프랑스의 수도는 어디인가요?')

# LangChain은 전달된 순서대로 모델에 전달하여 처리 
response = model.invoke([system_msg, human_msg])
print(response)