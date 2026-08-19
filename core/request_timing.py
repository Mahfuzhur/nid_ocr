import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from nid_ocr.core.config import settings


def _build_logger() -> logging.Logger:
    logger = logging.getLogger('nid_ocr.response_timing')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    logger.addHandler(console)

    log_file = settings.ocr_timing_log_file.strip()
    if log_file:
        try:
            path = Path(log_file)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path,
                maxBytes=settings.ocr_timing_log_max_bytes,
                backupCount=settings.ocr_timing_log_backup_count,
                encoding='utf-8',
            )
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning(f"Could not open OCR timing log file: {exc}")

    return logger


logger = _build_logger()


class OCRTimingMiddleware:
    """Record response latency for OCR endpoints without logging NID data."""

    _OCR_PATHS = {'/nid/front', '/nid/back'}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') not in self._OCR_PATHS:
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        state = scope.setdefault('state', {})
        state['request_id'] = uuid.uuid4().hex
        status_code = 500
        logged = False

        def log_once() -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            payload = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event': 'ocr_request',
                'request_id': state['request_id'],
                'method': scope.get('method'),
                'path': scope.get('path'),
                'status_code': status_code,
                'queue_wait_ms': state.get('queue_wait_ms'),
                'processing_ms': state.get('processing_ms'),
                'total_duration_ms': round((time.perf_counter() - started) * 1000, 2),
            }
            logger.info(json.dumps(payload, separators=(',', ':')))

        async def send_with_timing(message):
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message['status']
                headers = list(message.get('headers', []))
                headers.append((b'x-request-id', state['request_id'].encode('ascii')))
                message['headers'] = headers
            elif message['type'] == 'http.response.body' and not message.get('more_body', False):
                log_once()
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        except Exception:
            log_once()
            raise
        finally:
            log_once()
