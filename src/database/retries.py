"""Shared retry decorators for transient database errors."""

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from sqlalchemy.exc import (
    DisconnectionError,
    InterfaceError,
    OperationalError,
    TimeoutError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core.logs import safe_before_sleep_log

P = ParamSpec("P")
T = TypeVar("T")

# Transient errors that may succeed on retry.
# Deliberately excludes DBAPIError (parent of IntegrityError, etc.)
# so constraint violations fail fast instead of retrying 3x.
TRANSIENT_DB_EXCEPTIONS = (
    OperationalError,
    TimeoutError,
    DisconnectionError,
    InterfaceError,
)


def retry_on_transient_db_error(
    logger: logging.Logger,
    event_name: str,
    *,
    max_attempts: int = 3,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator factory for retrying on transient database errors.

    Args:
        logger: Logger instance for retry warnings.
        event_name: Event name for structured logging (e.g. "conversation.create.retry").
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        A decorator that wraps the function with retry logic.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(TRANSIENT_DB_EXCEPTIONS),
        wait=wait_exponential_jitter(),
        before_sleep=safe_before_sleep_log(logger, event_name),
        reraise=True,
    )
