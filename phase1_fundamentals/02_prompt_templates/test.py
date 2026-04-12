import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)


load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
PROVIDER = os.getenv("PROVIDER")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

model = init_chat_model(
    DEFAULT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    model_provider=PROVIDER,
    temperature=0.7,
    )

# template = PromptTemplate(
#     input_variables=["product", "feature"],
#     template="你是一个专业的产品描述员，负责描述 {product} 的 {feature} 功能。",
#    )
#
# prompt=template.format(product="苹果 17", feature="便宜")
# response = model.invoke(prompt)
# print(response.content)

# template2 = PromptTemplate.from_template("你是一个写作诗人，写出一首{theme}的诗词，像{people}那样的诗人风格一样。不超过4行。")
#
# prompt2 = template2.invoke({"theme": "爱情", "people": "李白"})
# response2 = model.invoke(prompt2)
# print(response2.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# # 创建组件
# template = ChatPromptTemplate.from_messages([
#     ("system", "你是{role}"),
#     ("user", "{input}")
# ])
#
# prompt = template.invoke({"role": "Python 导师", "input": "什么是装饰器？"})
# response = model.invoke(prompt)
# print(response.content)
#
# # 使用 | 创建链
# chain = template | model
#
# # 直接调用链
# response2 = chain.invoke({
#     "role": "Python 导师",
#     "input": "什么是装饰器？"
# })


# 3. from_template：创建模板对象
template = PromptTemplate.from_template(
    "你是一个诗人，请写一首关于{theme}的短诗，风格像{people}，不超过4行。"
)

print("===== 1) from_template 创建后的对象 =====")
print(template)
print(type(template))
print()

# 4. format：把变量填进去，返回普通字符串 str
prompt_text = template.format(theme="月亮", people="李白")

print("===== 2) format 返回的内容 =====")
print(prompt_text)
print(type(prompt_text))
print()

# 5. invoke：把变量填进去，返回 PromptValue 对象
prompt_value = template.invoke({"theme": "月亮", "people": "李白"})

print("===== 3) invoke 返回的内容 =====")
print(prompt_value)
print(type(prompt_value))
print()

# 如果想看 invoke 生成的纯文本，可以转成字符串
print("===== 4) invoke 转成字符串后 =====")
print(prompt_value.to_string())
print()

# 6. model.invoke：把提示词发给模型，得到 AI 回复
response = model.invoke(prompt_value)

print("===== 5) model.invoke 返回的内容 =====")
print(response)
print(type(response))
print()

print("===== 6) 模型真正回答的文本 =====")
print(response.content)