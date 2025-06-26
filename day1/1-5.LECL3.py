import asyncio # 비동기 라이브러리 
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

# 비동기 스트림을 사용하여 'YouTube' 토픽의 메시지 처리
async def main():
    # astream을 사용한 'YouTube' 토픽 처리
    print("--- 'YouTube' 토픽에 대한 비동기 스트리밍 시작 ---")
    print("모델이 토큰을 생성하는 대로 실시간으로 출력됩니다:\n")
    async for token in chain.astream({"topic": "YouTube"}):
        print(token, end="", flush=True) # flush -> 버퍼링 없이 
    print("\n--- 'YouTube' 스트리밍 종료 ---\n") # 구분자를 위한 줄 바꿈 추가

    # ainvoke를 사용한 'NVDA' 토픽 처리
    print("--- 'NVDA' 토픽에 대한 비동기 'ainvoke' 호출 시작 ---")
    # 비동기 체인 객체의 'ainvoke' 메서드를 호출합니다.
    my_process = chain.ainvoke({"topic": "NVDA"})

    print("비동기 작업 시작. 작업이 완료될 때까지 기다리는 중...")

    # 'await' 키워드를 사용하여 'my_process' 코루틴이 완료될 때까지 기다립니다.
    result = await my_process

    print("\n--- 'NVDA' 비동기 작업 완료 ---")
    print(f"모델 응답 (ainvoke): {result}")

if __name__ == "__main__":
  asyncio.run(main()) # asyncio.run()을 사용하여 비동기 함수 실행 

