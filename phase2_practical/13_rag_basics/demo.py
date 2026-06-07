"""
LangChain 1.0 - RAG Basics 演示（非交互式）
===========================================

快速演示所有 RAG 组件，无需按 Enter 确认
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec


# =========================================================
# 1. 路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. 环境变量与模型初始化
# =========================================================
def load_config():
    """加载环境变量配置"""
    load_dotenv()

    config = {
        "API_KEY": os.getenv("API_KEY"),
        "BASE_URL": os.getenv("BASE_URL"),
        "PROVIDER": os.getenv("PROVIDER"),
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    }
    return config


def create_chat_model(config):
    """创建聊天模型"""
    model = init_chat_model(
        config["DEFAULT_MODEL"],
        api_key=config["API_KEY"],
        base_url=config["BASE_URL"],
        model_provider=config["PROVIDER"]
    )
    return model


# =========================================================
# 3. 文档相关函数
# =========================================================
def prepare_sample_document():
    """创建示例文档并加载"""
    sample_text = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词模板
3. Chains - 链式调用
4. Agents - 智能代理

RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一。"""

    doc_path = DATA_DIR / "langchain_intro.txt"

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    return documents


def split_documents(documents):
    """对文档进行分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


# =========================================================
# 4. 向量模型相关函数
# =========================================================
def create_embeddings():
    """创建嵌入模型"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def test_embeddings(embeddings,query):
    """测试嵌入模型"""
    vector = embeddings.embed_query(query)
    return vector


# =========================================================
# 5. Pinecone 相关函数
# =========================================================
def create_pinecone_client(pinecone_api_key):
    """创建 Pinecone 客户端"""
    pc = Pinecone(api_key=pinecone_api_key)
    return pc


def get_or_create_index(pc, index_name, dimension):
    """获取或创建 Pinecone 索引"""
    existing_indexes = []

    for idx in pc.list_indexes():
        existing_indexes.append(idx.name)

    print("pc.list_indexes():", pc.list_indexes())
    print("existing_indexes:", existing_indexes)

    if index_name in existing_indexes:
        print("  [OK] 索引已存在")
        index = pc.Index(index_name)
    else:
        print("  创建新索引...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
        index = pc.Index(index_name)
        print("  [OK] 索引创建完成")

    return index


def create_vectorstore(chunks, embeddings, index_name):
    """将文档写入 Pinecone 向量库"""
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore


# =========================================================
# 6. 工具定义
# =========================================================
def create_search_tool(vectorstore):
    """创建知识库搜索工具"""

    @tool
    def search_knowledge_base(query: str) -> str:
        """在知识库中搜索相关信息"""
        docs = vectorstore.similarity_search(query, k=2)

        results = []
        for doc in docs:
            results.append(doc.page_content)

        return "\n\n".join(results)

    return search_knowledge_base


# =========================================================
# 7. Agent 相关函数
# =========================================================
def create_rag_agent(model, search_tool):
    """创建 RAG Agent"""
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt="你是一个助手，可以访问知识库。使用 search_knowledge_base 工具搜索相关信息，然后回答问题。"
    )
    return agent


def ask_question(agent, question):
    """向 Agent 提问"""
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )
    return response


# =========================================================
# 8. 主流程
# =========================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - RAG Basics 快速演示")
    print("=" * 70)

    # 读取配置
    config = load_config()

    # 创建模型
    model = create_chat_model(config)

    # [1/6] 文档加载
    print("\n[1/6] 文档加载...")
    documents = prepare_sample_document()
    print(f"  [OK] 加载了 {len(documents)} 个文档")

    # [2/6] 文本分割
    print("\n[2/6] 文本分割...")
    chunks = split_documents(documents)
    print(f"  [OK] 分割为 {len(chunks)} 个块")

    # [3/6] 向量嵌入
    print("\n[3/6] 向量嵌入 (首次运行会下载模型)...")
    embeddings = create_embeddings()
    vector = test_embeddings(embeddings,"LangChain 是什么")
    print(f"  [OK] 向量维度: {len(vector)}")

    # [4-6] Pinecone 相关
    if config["PINECONE_API_KEY"]:
        print("\n[4/6] Pinecone 设置...")
        try:
            pc = create_pinecone_client(config["PINECONE_API_KEY"])
            index_name = "langchain-rag-demo"
            dimension = 384

            index = get_or_create_index(pc, index_name, dimension)

            print("\n[5/6] 文档索引...")
            vectorstore = create_vectorstore(chunks, embeddings, index_name)
            print(f"  [OK] {len(chunks)} 个文档块已索引")

            print("\n[6/6] RAG 问答...")
            search_tool = create_search_tool(vectorstore)
            agent = create_rag_agent(model, search_tool)

            question = "LangChain 有哪些核心组件？"
            print(f"\n  问题: {question}")

            try:
                response = ask_question(agent, question)
                print(f"  回答: {response['messages'][-1].content}")
                print("\n  [OK] RAG 问答完成")
            except Exception as e:
                print(f"  [错误] RAG 问答失败: {e}")
                print("  提示: 可能是模型工具调用兼容性问题，不影响前面的 RAG 流程演示")

        except Exception as e:
            print(f"  [错误] Pinecone 操作失败: {e}")
    else:
        print("\n[4-6] 跳过 Pinecone 相关示例（未设置 API key）")

    print("\n" + "=" * 70)
    print(" 演示完成！")
    print("=" * 70)
    print("\n完整功能请运行: python main.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()