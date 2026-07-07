import os
import shutil
import tempfile
from dataclasses import asdict

from fastapi import APIRouter, UploadFile, File, HTTPException
from nid_ocr.api.schemas.response import NIDBackResponse
from nid_ocr.services.nid_back_service import NIDBackService
from nid_ocr.core.config import settings
from nid_ocr.core.exceptions import NIDOCRError
from nid_ocr.core.logging import get_logger
from nid_ocr.core.storage import save_upload
from nid_ocr.core.db import record_upload

logger = get_logger(__name__)


class NIDBackRouter:
    def __init__(self, service: NIDBackService):
        self._service = service
        self.router = APIRouter()
        self.router.add_api_route(
            "/nid/back",
            self._handle,
            methods=["POST"],
            response_model=NIDBackResponse,
            summary="Extract fields from NID back image",
        )

    async def _handle(
        self,
        file: UploadFile = File(...),
    ) -> NIDBackResponse:
        self._validate_extension(file.filename)

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        ocr = settings.default_ocr_engine
        stored_path = None
        try:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            stored_path = save_upload(tmp_path, "back", file.filename)

            logger.info(f"Processing back NID: {file.filename} (ocr={ocr})")
            result = self._service.process(tmp_path, ocr=ocr)
            record_upload("back", file.filename, str(stored_path) if stored_path else None, ocr, True, asdict(result))
            return NIDBackResponse(
                address=result.address,
                blood_group=result.blood_group,
                issue_date=result.issue_date,
                place_of_birth=result.place_of_birth,
            )
        except NIDOCRError as e:
            record_upload("back", file.filename, str(stored_path) if stored_path else None, ocr, False, error_message=str(e))
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error processing {file.filename}")
            record_upload("back", file.filename, str(stored_path) if stored_path else None, ocr, False, error_message="Internal processing error.")
            raise HTTPException(status_code=500, detail="Internal processing error.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _validate_extension(filename: str) -> None:
        ext = os.path.splitext(filename or '')[-1].lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions}",
            )
