import os
import time
from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec
from langchain.agents import create_agent

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
load_dotenv()
config = {
    "API_KEY": os.getenv("API_KEY"),
    "BASE_URL": os.getenv("BASE_URL"),
    "PROVIDER": os.getenv("PROVIDER"),
    "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
}


model = init_chat_model(
    config["DEFAULT_MODEL"],
    api_key=config["API_KEY"],
    base_url=config["BASE_URL"],
    model_provider=config["PROVIDER"]
)

sample_text = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词模板
3. Chains - 链式调用
4. Agents - 智能代理

RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

chunks = splitter.split_documents([Document(page_content=sample_text)])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [chunk.page_content for chunk in chunks]

# chunks_vector = embeddings.embed_documents(texts)
# query_vector = embeddings.embed_query("LangChain 是什么")

pc = Pinecone(api_key=config["PINECONE_API_KEY"])
index_name = "test2"

existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=config["PINECONE_API_KEY"]
)
similarity = vectorstore.similarity_search(query="LangChain 是什么",k = 2)


@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关内容"""
    docs = vectorstore.similarity_search(query=query, k=2)
    return "\n\n".join(doc.page_content for doc in docs)


agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="你是一个问答助手。回答问题前，先使用 search_knowledge_base 搜索资料，再根据资料回答。不要编造。"
)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "LangChain 有哪些核心组件？"}
        ]
    }
)

print(response["messages"][-1].content)