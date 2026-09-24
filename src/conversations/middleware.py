from collections.abc import Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

from conversations.schemas import LLMConfiguration

ALLOWED_MODELS = {
    "poolside/laguna-xs-2.1:free",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3.8-27b:free",
}


class DynamicModelMiddleware(AgentMiddleware[Any, LLMConfiguration]):
    def wrap_model_call(
        self,
        request: ModelRequest[LLMConfiguration],
        handler: Callable[[ModelRequest[LLMConfiguration]], ModelResponse],
    ) -> ModelResponse:
        from conversations.agent import get_chat_openrouter_client

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
