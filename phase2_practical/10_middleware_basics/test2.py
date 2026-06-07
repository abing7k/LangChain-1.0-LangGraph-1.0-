import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")



model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER)


class OutputValidationMiddleware(AgentMiddleware):
    """
    输出验证中间件 - 检查响应长度

    after_model 验证输出
    """

    def __init__(self, max_length=100):
        super().__init__()
        self.max_length = max_length

    def after_model(self, state, runtime):
        """模型响应后，验证输出"""
        messages = state.get('messages', [])
        if not messages:
            return None

        last_message = messages[-1]
        content = getattr(last_message, 'content', '')

        if len(content) > self.max_length:
            print(f"\n[警告] 响应时长 ({len(content)} 字符)，已截断到 {self.max_length}")
            # 这里可以实现截断或重试逻辑
            truncated_message = AIMessage(
                content=content[:self.max_length] + "...",
                id=getattr(last_message, "id", None),
            )

            return {"messages": [truncated_message]}

        return None


def example_4_output_validation():
    """
    示例4：输出验证 - 检查响应质量

    展示如何验证模型输出
    """
    print("\n" + "="*70)
    print("示例 4：输出验证 - 响应长度检查")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[OutputValidationMiddleware(max_length=50)]
    )

    print("\n用户: 请详细介绍 Python 编程语言的历史、特点和应用")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "请详细介绍 Python 编程语言的历史、特点和应用"}]
    })
    print(f"Agent: {response['messages'][-1].content}")

    print("\n关键点：")
    print("  - after_model 可以验证输出")
    print("  - 可以实现重试、截断等逻辑")
    print("  - 保证输出质量")


def main():
    example_4_output_validation()

if __name__ == '__main__':
    main()
