
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import os

from tools.web_search import web_search
from tools.calculator import calculator
from tools.weather import get_weather



load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER,)


model_with_tools   =  model.bind_tools([web_search, calculator, get_weather])

result   = model_with_tools.invoke("今天北京天气怎么样")

if result.tool_calls:
    print("模型想调用工具：")
    print(result.tool_calls)
    for tool_call in result.tool_calls:
        if tool_call["name"] == "get_weather":
            print(f"调用天气工具，参数： {tool_call}")
            weather_result = get_weather.invoke({'city': '上海'})
            print(f"调用天气工具，参数： {tool_call['args']}")
            print(weather_result)
else:
    print("模型没有调用工具")
    print(result.content)