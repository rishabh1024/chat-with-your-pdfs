import json
import logging
import logging.config
import logging.handlers
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from tenacity import RetryCallState

from core.settings import LoggingSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_CONFIGURATION_PATH = PROJECT_ROOT / "log_config.json"

_configuration_lock = Lock()
_is_configured = False


def _resolve_log_file_path(filename: str) -> Path:
    """Resolve a possibly-relative log filename against PROJECT_ROOT."""
    log_file = Path(filename)
    if not log_file.is_absolute():
        log_file = PROJECT_ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def configure_logger(settings: LoggingSettings | None = None) -> None:
    """Configure application logging once, using project-relative paths."""
    global _is_configured

    if _is_configured:
        return

    with _configuration_lock:
        if _is_configured:
            return

        logging_settings = settings or LoggingSettings()
        with LOG_CONFIGURATION_PATH.open(encoding="utf-8") as log_config:
            config = json.load(log_config)

        config["handlers"]["file"]["filename"] = str(
            _resolve_log_file_path(config["handlers"]["file"]["filename"])
        )

        logging.config.dictConfig(config)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging_settings.level)
        for handler in root_logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(logging_settings.level)

        _is_configured = True


def safe_before_sleep_log(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.WARNING,
) -> Callable[[RetryCallState], None]:
    """Return a Tenacity callback that omits exception messages and arguments."""

    def log_retry(retry_state: RetryCallState) -> None:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        logger.log(
            level,
            "%s attempt=%s error_type=%s",
            event,
            retry_state.attempt_number,
            type(exception).__name__ if exception else "unknown",
        )

    return log_retry
