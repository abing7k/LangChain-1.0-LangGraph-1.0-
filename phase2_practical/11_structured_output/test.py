import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

model = init_chat_model(
    DEFAULT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    model_provider=PROVIDER
)

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

agent = create_agent(
    model=model,
    tools=[],
    response_format=Person
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "张三是一名 30 岁的软件工程师"}
    ]
})

print(type(result))
print(result.keys())

person = result["structured_response"]
print(type(person))
print("姓名:", person.name)
print("年龄:", person.age)
print("职业:", person.occupation)