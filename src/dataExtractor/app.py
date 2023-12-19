import os
import datetime
import json
import traceback
import asyncio
import dotenv
import requests
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from celery.result import AsyncResult
from flask import Flask, request, render_template, send_from_directory, jsonify
from langchain import FAISS
from langchain import VectorDBQA, HuggingFaceHub, Cohere, OpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


if os.getenv("LLM_NAME") is not None:
    llm_choice = os.getenv("LLM_NAME")
else:
    llm_choice = "openai_chat"

if os.getenv("EMBEDDINGS_NAME") is not None:
    embeddings_choice = os.getenv("EMBEDDINGS_NAME")
else:
    embeddings_choice = "openai_text-embedding-ada-002"
    #embeddings_choice = "gpt-3.5-turbo"

if llm_choice == "manifest":
    from manifest import Manifest
    from langchain.llms.manifest import ManifestWrapper

    manifest = Manifest(
        client_name="huggingface",
        client_connection="http://127.0.0.1:5000"
    )

# Redirect PosixPath to WindowsPath on Windows
import platform

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# loading the .env file
dotenv.load_dotenv()

# load the prompts
with open("prompts/combine_prompt.txt", "r") as f:
    template = f.read()

with open("prompts/combine_prompt_hist.txt", "r") as f:
    template_hist = f.read()

with open("prompts/question_prompt.txt", "r") as f:
    template_quest = f.read()

with open("prompts/chat_combine_prompt.txt", "r") as f:
    chat_combine_template = f.read()

with open("prompts/chat_reduce_prompt.txt", "r") as f:
    chat_reduce_template = f.read()

if os.getenv("API_KEY") is not None:
    api_key_set = True
else:
    api_key_set = False
if os.getenv("EMBEDDINGS_KEY") is not None:
    embeddings_key_set = True
else:
    embeddings_key_set = False


async def async_generate(chain, question, chat_history):
    result = await chain.arun({"question": question, "chat_history": chat_history})
    return result

def run_async_chain(chain, question, chat_history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = {}
    try:
        answer = loop.run_until_complete(async_generate(chain, question, chat_history))
    finally:
        loop.close()
    result["answer"] = answer
    return result


def LLMCreater(llm_choice,docsearch,question):

    # create a prompt template
    if history:
        # print(f"History:{history}")
        history = json.loads(history)
        # print(f"History line no 184:{history}")
        template_temp = template_hist.replace("{historyquestion}", history['question']).replace("{historyanswer}",
                                                                                        history['answer'])
        c_prompt = PromptTemplate(input_variables=["summaries", "question","grades","category"], template=template_temp,
                                    template_format="jinja2")
    else:
        c_prompt = PromptTemplate(input_variables=["summaries", "question","grades","category"], template=template,
                                    template_format="jinja2")

    q_prompt = PromptTemplate(input_variables=["context", "question","grades","category"], template=template_quest,
                                template_format="jinja2")
    print("Passing the prompt to the model")
    if llm_choice == "openai_chat":

        llm = ChatOpenAI(openai_api_key=os.getenv("API_KEY"), model_name="gpt-4")
        
        messages_combine = [
            SystemMessagePromptTemplate.from_template(chat_combine_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        p_chat_combine = ChatPromptTemplate.from_messages(messages_combine)
        messages_reduce = [
            SystemMessagePromptTemplate.from_template(chat_reduce_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        p_chat_reduce = ChatPromptTemplate.from_messages(messages_reduce)
    elif llm_choice == "openai":
        llm = OpenAI(openai_api_key=os.getenv("API_KEY"), temperature=0)
    elif llm_choice == "manifest":
        llm = ManifestWrapper(client=manifest, llm_kwargs={"temperature": 0.001, "max_tokens": 2048})
    elif llm_choice == "huggingface":
        llm = HuggingFaceHub(repo_id="bigscience/bloom", huggingfacehub_api_token=os.getenv("API_KEY"))
    elif llm_choice == "cohere":
        llm = Cohere(model="command-xlarge-nightly", cohere_api_key=os.getenv("API_KEY"))

    if llm_choice == "openai_chat":
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm, chain_type="map_reduce", combine_prompt=p_chat_combine)
        chain = ConversationalRetrievalChain(
            retriever=docsearch.as_retriever(k=2),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
        )
        chat_history = []
        #result = chain({"question": question, "chat_history": chat_history})
        # generate async with async generate method
        result = run_async_chain(chain, question, chat_history)
    else:
        qa_chain = load_qa_chain(llm=llm, chain_type="map_reduce",
                                    combine_prompt=c_prompt, question_prompt=q_prompt)
        chain = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch, k=3)
        result = chain({"query": question})

    print(result)

    # some formatting for the frontend
    if "result" in result:
        result['answer'] = result['result']
    result['answer'] = result['answer'].replace("\\n", "\n")

    try:
        result['answer'] = result['answer'].split("SOURCES:")[0]
    except:
        pass


