import asyncio
import base64
import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from nid_ocr.api.schemas.response import NIDFrontResponse
from nid_ocr.core.concurrency import OCRBusyError, OCRConcurrencyGate
from nid_ocr.core.config import settings
from nid_ocr.core.exceptions import NIDOCRError
from nid_ocr.core.logging import get_logger
from nid_ocr.services.nid_front_service import NIDFrontService

logger = get_logger(__name__)


class NIDFrontRouter:
    def __init__(self, service: NIDFrontService, gate: OCRConcurrencyGate):
        self._service = service
        self._gate = gate
        self.router = APIRouter()
        self.router.add_api_route(
            "/nid/front",
            self._handle,
            methods=["POST"],
            response_model=NIDFrontResponse,
            summary="Extract fields from NID front image",
        )

    async def _handle(self, file: UploadFile = File(...)) -> NIDFrontResponse:
        self._validate_extension(file.filename)

        tmp_dir = tempfile.mkdtemp()
        safe_filename = os.path.basename((file.filename or 'upload').replace('\\', '/'))
        tmp_path = os.path.join(tmp_dir, safe_filename)
        ocr = settings.default_ocr_engine
        try:
            with open(tmp_path, "wb") as output:
                shutil.copyfileobj(file.file, output)

            logger.info(f"Processing front NID: {safe_filename} (ocr={ocr})")
            async with self._gate.slot():
                result, signature, bangla_name = await asyncio.to_thread(
                    self._service.process, tmp_path, ocr=ocr
                )
            return NIDFrontResponse(
                name=result.name,
                father_name=result.father_name,
                mother_name=result.mother_name,
                spouse_name=result.spouse_name,
                date_of_birth=result.date_of_birth,
                nid_number=result.nid_number,
                signature_base64=base64.b64encode(signature).decode() if signature else None,
                bangla_name=base64.b64encode(bangla_name).decode() if bangla_name else None,
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
            logger.exception(f"Unexpected error processing {safe_filename}")
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
