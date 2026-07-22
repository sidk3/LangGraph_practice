from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_token,
    streaming=True
)

chat = ChatHuggingFace(llm=llm)
def chat_node(state: ChatState):
    messages=state["messages"]
    response=chat.invoke(messages)
    return {'messages':[response]}


conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)


graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

def fetch_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
# CONFIG={'configurable':{'thread_id':'thread-2'}}
# response=chatbot.invoke(
#                 {'messages':[HumanMessage(content="What is my name?")]},
#                 config=CONFIG
#             )
# print(response)
