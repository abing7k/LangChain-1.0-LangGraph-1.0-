import os

from langchain.agents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")



model = init_chat_model(DEFAULT_MODEL,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        model_provider=PROVIDER)


path = "test2.db"
with SqliteSaver.from_conn_string(path) as checkpointer:
    agent = create_agent(model,
                         checkpointer=checkpointer,
                         middleware=[
                             SummarizationMiddleware(
                                 model=model,
                                 temperature=0.7,
                                 max_tokens=2000,
                             )
                         ])
    config = {"configurable": {"thread_id": "user_123"}}
    response = agent.invoke({"messages": [{"role": "user", "content": "你好,我叫什么"}]},
                           config=config)

    print(f"Agent: {response['messages'][-1].content}")
