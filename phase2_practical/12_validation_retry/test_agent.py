"""
LangChain 1.0 - Agent Validation + Retry + Fallbacks
====================================================

本示例演示：
1. create_agent() 创建 Agent
2. response_format 做结构化输出验证
3. ModelRetryMiddleware 做模型调用重试
4. ModelFallbackMiddleware 做模型降级
5. 外层重试循环处理业务验证失败
"""

import os
from dotenv import load_dotenv
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ValidationError

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
)

load_dotenv()

# ============================================================================
# 模型初始化
# ============================================================================
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

API_KEY2 = os.getenv("API_KEY2")
BASE_URL2 = os.getenv("BASE_URL2")
PROVIDER2 = os.getenv("PROVIDER2")
DEFAULT_MODEL2 = os.getenv("DEFAULT_MODEL2")

model = init_chat_model(
    DEFAULT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    model_provider=PROVIDER
)

model2 = init_chat_model(
    DEFAULT_MODEL2,
    api_key=API_KEY2,
    base_url=BASE_URL2,
    model_provider=PROVIDER2
)

# ============================================================================
# Pydantic 结构化输出模型（验证）
# ============================================================================
class ExtractedData(BaseModel):
    """提取的数据"""
    name: str = Field(description="名称", min_length=1)
    value: float = Field(description="数值，必须是正数", gt=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name 不能为空")
        if v.lower() == "unknown":
            raise ValueError("name 不能是 unknown")
        return v


def business_validate(data: ExtractedData) -> None:
    """
    业务级验证（Pydantic 之外的验证）
    不通过时直接抛异常，交给外层循环处理
    """
    # 这里纯演示：假设我们要求 value 至少 > 10
    if data.value <= 10:
        raise ValueError("业务验证失败：value 必须大于 10")


# ============================================================================
# 创建 Agent：结构化输出 + 模型重试 + 模型降级
# ============================================================================
def build_robust_agent():
    """
    Agent 版组合策略：

    1. response_format=ExtractedData  -> 输出验证
    2. ModelRetryMiddleware          -> 模型调用失败自动重试
    3. ModelFallbackMiddleware       -> 主模型失败自动降级
    """
    agent = create_agent(
        model=model,
        tools=[],  # 这里先不放工具，专注演示结构化输出
        response_format=ExtractedData,
        middleware=[
            ModelRetryMiddleware(
                max_retries=2,          # 失败后最多再试 2 次（总共最多 3 次）
                backoff_factor=2.0,     # 指数退避
                initial_delay=1.0,      # 初始等待 1 秒
                max_delay=10.0,         # 最大等待 10 秒
                jitter=True,            # 随机抖动
                retry_on=(ConnectionError, TimeoutError),
                on_failure="error",     # 全部失败后抛异常
            ),
            ModelFallbackMiddleware(
                model2  # 主模型失败后切换到备用模型
            )
        ],
    )
    return agent


# ============================================================================
# 外层工作流：处理业务验证失败时的重试
# ============================================================================
def extract_with_agent_validation(text: str, max_retries: int = 3) -> Optional[ExtractedData]:
    """
    使用 Agent 提取结构化数据，并加入外层业务重试逻辑

    说明：
    - Agent 内部负责：模型重试 + 模型降级 + 结构化输出
    - 外层循环负责：业务验证失败时的再提示再重试
    """
    agent = build_robust_agent()

    current_prompt = f"""请从下面文本中提取信息。

要求：
1. 返回字段：name, value
2. value 必须是数字类型（float/int 都可以）
3. 不要返回 unknown 作为 name

文本：
{text}
"""

    for attempt in range(1, max_retries + 1):
        print(f"\n尝试 {attempt}/{max_retries}")

        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": current_prompt}
                ]
            })

            # create_agent + response_format 的结构化结果
            data = result["structured_response"]

            # 额外业务验证
            business_validate(data)

            print("✓ 提取成功")
            print(f"  name : {data.name}")
            print(f"  value: {data.value}")
            return data

        except ValidationError as e:
            # Pydantic 字段验证失败
            print(f"✗ Pydantic 验证失败: {e}")
            if attempt < max_retries:
                current_prompt += (
                    f"\n\n上一次输出未通过字段验证：{str(e)}"
                    f"\n请严格按照 schema 返回，并确保类型正确。"
                )
            else:
                print("→ 达到最大重试次数")
                return None

        except ValueError as e:
            # 业务验证失败
            print(f"✗ 业务验证失败: {e}")
            if attempt < max_retries:
                current_prompt += (
                    f"\n\n上一次结果未通过业务规则：{str(e)}"
                    f"\n请修正后重新输出。"
                )
            else:
                print("→ 达到最大重试次数")
                return None

        except Exception as e:
            # 模型调用失败、fallback 后仍失败等
            print(f"✗ Agent 调用失败: {e}")
            if attempt < max_retries:
                current_prompt += (
                    "\n\n请注意：value 必须是数字类型，name 不能为空，"
                    "并且不要输出 unknown。请重新提取。"
                )
            else:
                print("→ 达到最大重试次数")
                return None

    return None


# ============================================================================
# 示例运行
# ============================================================================
def example_7_agent_combined():
    print("\n" + "=" * 70)
    print("示例 7：Agent 组合策略 - retry + fallbacks + validation")
    print("=" * 70)

    test_cases = [
        "产品 A 的价值是 999.99 元",
        "产品 B 的价值是 8 元",
        "unknown 的价值是 120 元",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"输入文本: {text}")
        result = extract_with_agent_validation(text, max_retries=3)

        if result:
            print("最终结果: 成功")
        else:
            print("最终结果: 失败")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    example_7_agent_combined()