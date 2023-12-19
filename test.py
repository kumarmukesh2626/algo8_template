import os
from autogen import config_list_from_json
import autogen
import json
import asyncio

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Define research function
import openai

os.environ["OPENAI_API_KEY"] = "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"
llm_config = {"config_list": config_list, "seed": 42}

# Set your OpenAI API key

user_proxy = autogen.UserProxyAgent(
    name="admin",
    system_message="you are the admin ",
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
)

solver = autogen.AssistantAgent(
    name="solver",
    system_message="your work is to answer the query received by user proxy agent ",
    llm_config={"config_list": config_list},
)

questioner = autogen.AssistantAgent(
    name="questioner",
    system_message="your work is to generate some questions not more than 5  based on the solver agent ",
    llm_config={"config_list": config_list},
)

ender = autogen.AssistantAgent(
    name="ender",
    system_message="your work is to present the text given by solver and questioner agent",
    llm_config={"config_list": config_list},
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, solver, questioner, ender],
    messages=[]
)

async def run_agents():
    for agent in groupchat.agents:
        agent.reset()
        await agent.initiate_chat(manager, message="tell me about up")
        await asyncio.sleep(1)  # Add a delay between agent interactions

loop = asyncio.get_event_loop()
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
loop.run_until_complete(run_agents())
