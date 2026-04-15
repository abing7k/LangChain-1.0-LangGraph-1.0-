"""
简单测试：验证 Agent 执行循环
"""

import os
import sys

# 添加工具目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from calculator import calculator

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")



if not API_KEY or API_KEY == "your_API_KEY_here_replace_this":
    raise ValueError("请先设置 API_KEY")

model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER,)

print("=" * 70)
print("测试：Agent 执行循环")
print("=" * 70)

agent = create_agent(model=model, tools=[calculator])

print("\n问题：10 加 20 等于多少？")
response = agent.invoke({
    "messages": [{"role": "user", "content": "10 加 20 等于多少？"}]
})

print("\n完整消息历史：")
for i, msg in enumerate(response['messages'], 1):
    msg_type = msg.__class__.__name__
    print(f"\n消息 {i}: {msg_type}")

    if hasattr(msg, 'content') and msg.content:
        print(f"  内容: {msg.content}")

    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  工具调用: {msg.tool_calls[0]['name']}")

print("\n" + "=" * 70)
print("最终答案:", response['messages'][-1].content)
print("=" * 70)

# 测试流式输出
print("\n测试流式输出：")
print("问题：5 乘以 6")
print("-" * 70)

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "5 乘以 6"}]
}):
    if 'messages' in chunk:
        latest = chunk['messages'][-1]
        if hasattr(latest, 'content') and latest.content:
            if not hasattr(latest, 'tool_calls') or not latest.tool_calls:
                print(f"最终答案: {latest.content}")

print("\n测试成功！")
