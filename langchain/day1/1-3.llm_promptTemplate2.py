# ChatPromptTemplate은 여러 메시지 타입을 조합해서 템플릿을 만듬
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(
  model = "gemma3:latest",
  base_url = "http://localhost:11434",
  temperature=0.7
)

# 심플한 ChatPromptTemplate;
# 이 문자열이 패워진 단일 HumanMessage 객체 리스트로 반환됨. 
chat_template_simple = ChatPromptTemplate.from_template(
  "너는 친절한 어시스턴트야. 질문에 대답하면 돼 :{question}"
)
final_chat_prompt_simple = chat_template_simple.invoke({"question": "프랑스의 수도는 어디야?"})

response_chat_simple = model.invoke(final_chat_prompt_simple)
print(f"모델 응답: {response_chat_simple.content}")
print("-" * 50)

# ChatPromptTemplate를 통해 여러 메세지 타입 조합하여 템플릿 생성
# 복잡하게 구성하기 위해서는 from_messages 활용; role 명시 가능 
chat_template = ChatPromptTemplate.from_messages([
  ('system','''아래 작성한 컨텍스트(Context)를 기반으로
   질문(Question)에 대답하세요. 제공된 정보로 대답할 수 없는 질문이라면 "모르겠어요."라고 답하세요.'''),
  ('human', 'Context: {context}'),
  ('human','Question: {question}'),
])

prompt = chat_template.invoke({
  'context':'''거대 언어 모델(LLM)은 자연어 처리(NLP)분야의 최신 발전을 이끌고 있습니다.
  거대 언어 모델은 더 작은 모델보다 우수한 성능을 보이며
  NLP 기능을 갖춘 애플리케이션을 개발하는 개발자들에게
  매우 중요한 도구가 되었습니다. 개발자들은 Hugging Face의 'transformers'라이브러리를
  활용하거나, 'openai' 및 'cohere' 라이브러리를 통해 OpenAI와 Cohere 서비스를 이용하여
  거대 언어 모델을 활용할 수 있습니다.''',
  'question':"거대 언어 모델은 어디서 제공하나요?"
})

completion = model.invoke(prompt)

print(completion)