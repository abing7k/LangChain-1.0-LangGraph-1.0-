
import os
from http.client import responses

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")



model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER)


agent = create_agent(model,
                     checkpointer=InMemorySaver()
                     )
config = {"configurable": {"thread_id": "conversation_1"}}

config2 = {"configurable": {"thread_id": "conversation_2"}}

response = agent.invoke({"messages": [{"role": "user", "content": "你好, 我是张三"}]}, config=config)
print(response["messages"][-1].content)
response2 = agent.invoke({"messages": [{"role": "user", "content": "我是谁？"}]}, config=config)
print(response2["messages"][-1].content)


response = agent.invoke({"messages": [{"role": "user", "content": "你好, 我是李四"}]}, config=config2)
print(response["messages"][-1].content)
response2 = agent.invoke({"messages": [{"role": "user", "content": "我是谁？"}]}, config=config2)
print(response2["messages"][-1].content)
print(response2)
