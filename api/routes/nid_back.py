import asyncio
import os
import shutil
import tempfile
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from nid_ocr.api.schemas.response import NIDBackResponse
from nid_ocr.core.concurrency import OCRBusyError, OCRConcurrencyGate
from nid_ocr.core.config import settings
from nid_ocr.core.exceptions import NIDOCRError
from nid_ocr.core.logging import get_logger
from nid_ocr.services.nid_back_service import NIDBackService

logger = get_logger(__name__)


class NIDBackRouter:
    def __init__(self, service: NIDBackService, gate: OCRConcurrencyGate):
        self._service = service
        self._gate = gate
        self.router = APIRouter()
        self.router.add_api_route(
            "/nid/back",
            self._handle,
            methods=["POST"],
            response_model=NIDBackResponse,
            summary="Extract fields from NID back image",
        )

    async def _handle(self, request: Request, file: UploadFile = File(...)) -> NIDBackResponse:
        self._validate_extension(file.filename)

        tmp_dir = tempfile.mkdtemp()
        safe_filename = os.path.basename((file.filename or 'upload').replace('\\', '/'))
        tmp_path = os.path.join(tmp_dir, safe_filename)
        ocr = settings.default_ocr_engine
        try:
            with open(tmp_path, "wb") as output:
                shutil.copyfileobj(file.file, output)

            logger.info(f"Processing back NID (ocr={ocr})")
            queue_started = time.perf_counter()
            try:
                async with self._gate.slot():
                    request.state.queue_wait_ms = round((time.perf_counter() - queue_started) * 1000, 2)
                    processing_started = time.perf_counter()
                    try:
                        result = await asyncio.to_thread(self._service.process, tmp_path, ocr=ocr)
                    finally:
                        request.state.processing_ms = round(
                            (time.perf_counter() - processing_started) * 1000, 2
                        )
            except OCRBusyError:
                request.state.queue_wait_ms = round((time.perf_counter() - queue_started) * 1000, 2)
                raise
            return NIDBackResponse(
                address=result.address,
                blood_group=result.blood_group,
                issue_date=result.issue_date,
                place_of_birth=result.place_of_birth,
            )
        except OCRBusyError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={'Retry-After': str(max(1, int(settings.ocr_queue_wait_seconds)))},
            ) from exc
        except NIDOCRError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"Unexpected back OCR error ({type(exc).__name__})")
            raise HTTPException(status_code=500, detail="Internal processing error.") from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _validate_extension(filename: str | None) -> None:
        ext = os.path.splitext(filename or '')[-1].lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions}",
            )
