import asyncio


class IndexingStatusTracker:
    """In-memory tracker for background indexing tasks.
    Each document_id gets an asyncio.Event that fires once indexing completes or fails.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._results: dict[str, dict[str, str]] = {}

    def register(self, document_id: str) -> None:
        self._events[document_id] = asyncio.Event()

    def mark_complete(self, document_id: str, status: str, message: str) -> None:
        self._results[document_id] = {"status": status, "message": message}
        if event := self._events.get(document_id):
            event.set()

    async def wait_for_result(
        self, document_id: str, timeout: float = 300.0
    ) -> dict[str, str]:
        event = self._events.get(document_id)
        if event is None:
            return {
                "status": "not_found",
                "message": "No indexing task registered for this document.",
            }
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            return {
                "status": "timeout",
                "message": "Indexing did not complete within the timeout period.",
            }
        return self._results.get(
            document_id, {"status": "unknown", "message": "Result missing."}
        )

    def cleanup(self, document_id: str) -> None:
        self._events.pop(document_id, None)
        self._results.pop(document_id, None)
