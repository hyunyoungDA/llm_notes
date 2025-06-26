from langchain_community.llms import Ollama
# from langchain_community.llms import OllamaLLM

llm_ollama = Ollama(
  model = "gemma3",
  base_url="http://localhost:11434",
  temperature = 0.7,
)

response_ollama = llm_ollama.invoke("안녕하세요? 오늘 날씨는 어떤가요?")
print(response_ollama)