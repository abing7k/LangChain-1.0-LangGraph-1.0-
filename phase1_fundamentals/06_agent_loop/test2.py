from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

import ast
import operator
from langchain_core.tools import tool

import os
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")




# 允许的运算符
ALLOWED_OPERATORS = {
    ast.Add: operator.add,       # +
    ast.Sub: operator.sub,       # -
    ast.Mult: operator.mul,      # *
    ast.Div: operator.truediv,   # /
    ast.Pow: operator.pow,       # **
    ast.USub: operator.neg,      # -x
    ast.UAdd: operator.pos,      # +x
}


import ast
import operator
from langchain_core.tools import tool

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_calculate(expression: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只允许数字")

        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")

            return ALLOWED_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")

            return ALLOWED_OPERATORS[op_type](operand)

        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        return _eval(tree)
    except ZeroDivisionError:
        raise ValueError("除数不能为 0")
    except Exception as e:
        raise ValueError(f"表达式错误: {e}")

@tool
def get_calculator(input: str) -> str:
    """
    计算数学表达式，例如：
    - 100 / 5
    - 2 + 3 * 4
    - (10 - 2) / 4
    """
    try:
        result = safe_calculate(input)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算失败: {e}"

@tool
def calculator(input: str) -> str:
    """
    计算数学表达式。
    输入应该是一个数学表达式字符串，例如:
    '100 / 5'
    '2 + 3 * 4'
    '(10 - 2) / 4'
    """
    try:
        result = safe_calculate(input)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算失败: {e}"

model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER,)




print("\n" + "=" * 70)
print("示例 4：查看中间状态")
print("=" * 70)

agent = create_agent(
    model=model,
    tools=[calculator]
)

print("\n问题：100 除以 5 等于多少？")
print("\n执行步骤：")

step = 0

# 关键：
# stream_mode="updates" -> 看每一步更新
# version="v2" -> 返回统一结构：type / ns / data
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "100 除以 5 等于多少？"}
        ]
    },
    stream_mode="updates",
    version="v2",
):
    # 只处理 updates 类型
    if chunk.get("type") != "updates":
        continue

    data = chunk.get("data", {})

    # data 通常是：
    # {
    #   "model": {"messages": [...]}
    # }
    # 或
    # {
    #   "tools": {"messages": [...]}
    # }
    for node_name, state in data.items():
        step += 1
        print(f"\n步骤 {step}:")
        print(f"  节点: {node_name}")

        if "messages" not in state or not state["messages"]:
            print("  没有 messages")
            continue

        latest = state["messages"][-1]
        msg_type = latest.__class__.__name__
        print(f"  类型: {msg_type}")

        # 1) 模型准备调用工具
        if hasattr(latest, "tool_calls") and latest.tool_calls:
            tool_call = latest.tool_calls[0]
            tool_name = tool_call.get("name", "未知工具")
            tool_args = tool_call.get("args", {})
            print(f"  工具调用: {tool_name}")
            print(f"  工具参数: {tool_args}")

        # 2) 工具执行结果 / 3) 模型最终回答
        elif hasattr(latest, "content") and latest.content:
            content = str(latest.content).strip()
            print(f"  内容: {content}")

        else:
            print("  内容为空")

print("\n关键点：")
print("  - stream 让你看到每个步骤")
print("  - updates 模式适合看 Agent 执行过程")
print("  - version='v2' 的 chunk 结构更统一")
print("  - 可以用于调试")
print("  - 可以用于进度显示")