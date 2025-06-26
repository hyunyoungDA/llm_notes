from langchain_core.prompts import PromptTemplate # 주로 단일 문자열 형태의 프롬프트 생성 시 사용 
from langchain_core.prompts import ChatMessagePromptTemplate # 채팅 애플리케이션 구축 시 유용함 
from langchain_ollama import ChatOllama

model = ChatOllama(
  model = "gemma3:latest",
  base_url = "http://localhost:11434",
  temperature=0.7,
)

# from_template: 미리 정의된 문자열 템플릿으로부터 프롬프트 객체를 생성하는 역할 
# 입력된 템플릿 문자열 안에 중괄호 구문이 있으면 템플릿 변수로 자동 인식 -> 실제로 채워질 슬롯(플레이스홀더) 역할 
template = PromptTemplate.from_template('''아래 작성한 컨텍스트(Context)를 기반으로 질문(Question)에 대답하세요. 
    제공된 정보로 대답할 수 없는 질문이라면 '모르겠어요'라고 답하세요
                                        
Context: {context} 

Question: {question}

Answer:''')

prompt = template.invoke({
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