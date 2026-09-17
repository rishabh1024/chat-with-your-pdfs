from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.types import AgentState, InputAgentState, OutputAgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr
from src.conversations.prompts import AGENT_SYSTEM_PROMPT
from src.conversations.protocols import DocumentVectorStore
from src.conversations.schemas import LLMConfiguration
from src.conversations.tools import MongoDocumentSearch, build_search_documents_tool
from src.core.settings import load_environment_variables

settings = load_environment_variables()
OPENROUTER_API_KEY = settings.openrouter.openrouter_api_key.get_secret_value()


_LLM_DEFAULT_CONFIG = LLMConfiguration()


@lru_cache(maxsize=16)
def get_chat_openrouter_client(
    model_name: str = _LLM_DEFAULT_CONFIG.model_name,
    temperature: float = _LLM_DEFAULT_CONFIG.temperature,
    top_p: float = _LLM_DEFAULT_CONFIG.top_p,
    frequency_penalty: float = _LLM_DEFAULT_CONFIG.frequency_penalty,
    max_tokens: int = _LLM_DEFAULT_CONFIG.max_tokens,
    seed: int = _LLM_DEFAULT_CONFIG.seed,
) -> BaseChatModel:
    # Lazy import avoids circular dependency with middleware (ALLOWED_MODELS lives there).
    from src.conversations.middleware import ALLOWED_MODELS

    return ChatOpenRouter(
        model=model_name,
        api_key=SecretStr(OPENROUTER_API_KEY or ""),
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
        seed=seed,
        model_kwargs={"models": list(ALLOWED_MODELS)},
    )


# Imported after factory so middleware can import this module without a cycle.
from src.conversations.middleware import DynamicModelMiddleware  # noqa: E402


class RAGAgent:
    def __init__(
        self,
        model_configuration: LLMConfiguration,
        checkpointer: PostgresSaver,
        mongo_db_document_search: DocumentVectorStore | None = None,
    ) -> None:
        self.model_configuration = model_configuration
        self.checkpointer = checkpointer
        self.mongo_db_document_search = mongo_db_document_search or MongoDocumentSearch()
        self._rag_agent_with_tools = self._create_rag_agent()

    def _create_rag_agent(
        self,
    ) -> CompiledStateGraph[
        AgentState[Any], LLMConfiguration, InputAgentState, OutputAgentState[Any]
    ]:
        default_model = get_chat_openrouter_client(
            model_name=self.model_configuration.model_name,
            temperature=self.model_configuration.temperature,
            top_p=self.model_configuration.top_p,
            frequency_penalty=self.model_configuration.frequency_penalty,
            max_tokens=self.model_configuration.max_tokens,
            seed=self.model_configuration.seed,
        )
        return create_agent(
            model=default_model,
            tools=[build_search_documents_tool(self.mongo_db_document_search)],
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            context_schema=LLMConfiguration,
            middleware=[
                DynamicModelMiddleware(),
                ToolRetryMiddleware(backoff_factor=2.0, initial_delay=1.0),
                ModelRetryMiddleware(backoff_factor=2.0, initial_delay=1.0),
            ],
            name="RAG Agent",
        )

    @property
    def agent_graph(
        self,
    ) -> CompiledStateGraph[
        AgentState[Any], LLMConfiguration, InputAgentState, OutputAgentState[Any]
    ]:
        return self._rag_agent_with_tools
