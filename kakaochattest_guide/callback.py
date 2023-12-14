from dto import ChatbotRequest
import requests
import logging
import openai
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import langchain
langchain.debug = True
# 환경 변수 처리 필요!
openai.api_key = os.environ["GPT_KEY"]
os.environ["OPENAI_API_KEY"] = os.environ["GPT_KEY"]
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

raw_documents = TextLoader('./data/project_data_카카오싱크.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
DB = Chroma.from_documents(documents, OpenAIEmbeddings())


def callback_handler(request: ChatbotRequest) -> dict:

    # ===================== start =================================

    llm_chain = generate_langchain()
    response = llm_chain.run(request.userRequest.utterance)
    output_text = response

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url, json=payload)


def generate_tools(db):
    tools = [
        Tool(
            name="Search",
            func=db.similarity_search,
            description="카카오톡싱크에 대해서 관련된 문장을 검색하기 위해서 사용"
        ),
    ]
    return tools


def generate_langchain():
    tools = generate_tools(DB)
    llm = ChatOpenAI(temperature=0.8, model_name='gpt-4') # gpt-3.5-turbo는 generic한 answer에 parse error를 발생시킴, https://github.com/langchain-ai/langchain/issues/1358
    mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return mrkl