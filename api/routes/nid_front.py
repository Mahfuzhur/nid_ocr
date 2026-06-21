import os
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from nid_ocr.api.schemas.response import NIDFrontResponse
from nid_ocr.services.nid_front_service import NIDFrontService
from nid_ocr.core.config import settings
from nid_ocr.core.exceptions import NIDOCRError
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class NIDFrontRouter:
    def __init__(self, service: NIDFrontService):
        self._service = service
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
        tmp_path = os.path.join(tmp_dir, file.filename)
        try:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            logger.info(f"Processing front NID: {file.filename}")
            result = self._service.process(tmp_path)
            return NIDFrontResponse(
                name=result.name,
                father_name=result.father_name,
                mother_name=result.mother_name,
                spouse_name=result.spouse_name,
                date_of_birth=result.date_of_birth,
                nid_number=result.nid_number,
            )
        except NIDOCRError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error processing {file.filename}")
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
