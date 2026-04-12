import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")

model = init_chat_model(
    DEFAULT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    model_provider=PROVIDER
)

def chat():
    message = [
        {"role": "system", "content": "你是一个论文写作助手，表达自然、结构清晰"},
        {"role": "user", "content": "你好，关于Spring Boot的洗衣机店管理项目的摘要。"}
    ]

    while True:
        response = model.invoke(message)
        message.append({"role": "assistant", "content": response.content})
        print(response.content)

        user_input = input("请修意见：")
        if user_input.strip().lower() == "quit":
            break

        message.append({"role": "user", "content": user_input})

if __name__ == "__main__":
    chat()