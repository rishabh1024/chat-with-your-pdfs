from collections.abc import Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from src.conversations.schemas import LLMConfiguration

ALLOWED_MODELS = {
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "poolside/laguna-xs-2.1:free",
    "meta-llama/llama-3.2-3b-instruct:free",
}


class DynamicModelMiddleware(AgentMiddleware[Any, LLMConfiguration]):
    def wrap_model_call(
        self,
        request: ModelRequest[LLMConfiguration],
        handler: Callable[[ModelRequest[LLMConfiguration]], ModelResponse],
    ) -> ModelResponse:
        from src.conversations.agent import get_chat_openrouter_client

        ctx = request.runtime.context
        if ctx is None or ctx.model_name not in ALLOWED_MODELS:
            return handler(request)

        model = get_chat_openrouter_client(
            model_name=ctx.model_name,
            temperature=ctx.temperature,
            top_p=ctx.top_p,
            frequency_penalty=ctx.frequency_penalty,
            max_tokens=ctx.max_tokens,
            seed=ctx.seed,
        )
        return handler(request.override(model=model))
