import asyncio
import json
import os 

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv("../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#print(OPENAI_API_KEY)
async def main():
    
    with open("mcp_config.json") as f: # config.json 
        config = json.load(f)

    
    client = MultiServerMCPClient(config["mcpServers"]) # MCP 서버 등록

    # Tool 로딩 (Node 서버를 stdio로 실행) -> 따로 mcp server 활성화 필요 X 
    # streamable http 타입인 경우 다른 터미널에서 node run build로 mcp server 활성화 
    tools = await client.get_tools()

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = OPENAI_API_KEY)

    agent = create_react_agent(model.bind_tools(tools), tools) # ReAct

    
    query = "PyTorch로 인공신경망 예제 코드를 찾아서 만들어줘. 참고 문서 링크도 포함."
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    print(result["messages"][-1].content) # 가장 최근 messages의 content만 추출 

if __name__ == "__main__":
    asyncio.run(main())
