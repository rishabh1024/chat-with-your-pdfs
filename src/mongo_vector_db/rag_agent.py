import os

# from dotenv import load_dotenv
from langchain.agents.factory import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig
from pydantic import SecretStr

from src.mongo_vector_db.data_wrangler import DocumentIndexer

# load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

@tool(description="This tool searches for relevant data in the vectorDB based on the search query")
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for information relevant to the query."""
    print(f"Tool has been called with query: {query}")

    documents = DocumentIndexer.get_similar_documents_from_database(user_query=query)
    if isinstance(documents, list):
        return "\n\n---\n\n".join(document.page_content for document in documents)
    return f"No relevant document's were found. {documents['message']}"

llm = ChatOpenRouter(
    model="qwen/qwen3-30b-a3b-instruct-2507",
    api_key=SecretStr(OPENROUTER_API_KEY or ""),
    model_kwargs={
        "models": [
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "poolside/laguna-xs-2.1:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]
    },
    reasoning={'effort': 'medium'}
)

system_prompt = SystemMessage(
    content="You are a helpful assistant. Answer the user's questions clearly and concisely. "
    "If the question is about the Rishabh or you need more information to answer"
    "your questions, use the search_documents tool to get information from relevant documents. "
    "Otherwise answer directly. Do not fabricate information when asked about Rishabh."
)

agent = create_agent(
    model=llm,
    tools=[search_documents],
    checkpointer=InMemorySaver(),
)

config = RunnableConfig({"configurable": {"thread_id": str(uuid7())}})

result = agent.invoke(
    {"messages": [{"role": "user",
                   "content": """Can you share some insights about """
                   """Rishabh's educational background"""}]},
    config=config,
)

for message in (result['messages']):
  if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
    print(message.content)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What was his most recent experience?"}]},
    config=config,
)

for message in result["messages"]:
    if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
        print(message.content)