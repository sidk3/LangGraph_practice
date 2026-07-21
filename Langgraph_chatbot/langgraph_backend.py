from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

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

checkpointer=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

# print("HI")
# print(chatbot.invoke({'messages': [HumanMessage(content="What is use of Langgraph?")]}, 
#                 config={'configurable':{'thread_id':"thread-1"}},)['messages'][-1].content)
# print("BYE")
# for message_chunk,metadata in chatbot.invoke(
#     {'messages': [HumanMessage(content="What is use of Langgraph?")]},
#     config={'configurable':{'thread_id':"thread-1"}},
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content," ",flush=True)


