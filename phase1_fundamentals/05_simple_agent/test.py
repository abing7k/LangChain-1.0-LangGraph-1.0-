import os
import sys

from langchain_core.messages import HumanMessage

# 添加父目录到路径以导入工具
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # LangChain 1.0 统一 API

# 导入自定义工具
from weather import get_weather
from calculator import calculator
from web_search import web_search

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER,)


agent = create_agent(model,
                     tools=[calculator])


humanMessage = HumanMessage(content="北京今天天气怎么样？")

# response = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "北京今天天气怎么样？"}
#     ]
# })
#
# print(response["messages"][-1].content)

response = agent.invoke({
    "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
})
#
# # 显示完整的消息历史
# print("\n完整消息历史：")
# for i, msg in enumerate(response['messages'], 1):
#     print(f"\n--- 消息 {i} ({msg.__class__.__name__}) ---")
#     if hasattr(msg, 'content'):
#         print(f"内容：{msg.content}")
#     if hasattr(msg, 'tool_calls') and msg.tool_calls:
#         print(f"工具调用：{msg.tool_calls}")

response = agent.invoke({"messages": [{"role": "user", "content": "你好，你是什么模型"}]})
print(response)
print(response["messages"][-1].content)