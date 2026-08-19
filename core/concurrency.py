import asyncio
from contextlib import asynccontextmanager


class OCRBusyError(Exception):
    """Raised when all GPU inference slots remain busy past the queue limit."""


class OCRConcurrencyGate:
    def __init__(self, limit: int, wait_seconds: float):
        if limit < 1:
            raise ValueError("OCR concurrency limit must be at least 1")
        self._semaphore = asyncio.Semaphore(limit)
        self._wait_seconds = wait_seconds

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._wait_seconds)
        except TimeoutError as exc:
            raise OCRBusyError("OCR service is busy; retry shortly.") from exc
        try:
            yield
        finally:
            self._semaphore.release()
