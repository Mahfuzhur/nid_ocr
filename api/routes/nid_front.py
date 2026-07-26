import base64
import os
import shutil
import tempfile
from dataclasses import asdict

from fastapi import APIRouter, UploadFile, File, HTTPException
from nid_ocr.api.schemas.response import NIDFrontResponse
from nid_ocr.services.nid_front_service import NIDFrontService
from nid_ocr.core.config import settings
from nid_ocr.core.exceptions import NIDOCRError
from nid_ocr.core.logging import get_logger
from nid_ocr.core.storage import save_upload, save_bytes
from nid_ocr.core.db import record_upload

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

    async def _handle(
        self,
        file: UploadFile = File(...),
    ) -> NIDFrontResponse:
        self._validate_extension(file.filename)

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        ocr = settings.default_ocr_engine
        stored_path = None
        try:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            stored_path = save_upload(tmp_path, "front", file.filename)

            logger.info(f"Processing front NID: {file.filename} (ocr={ocr})")
            result, signature, bangla_name = self._service.process(tmp_path, ocr=ocr)
            # The signature/Bangla-name crops are each saved as their own file
            # (like the original upload) and their paths recorded separately
            # from extracted_data — keeping the base64 blobs out of that JSON
            # column avoids bloating the DB row and dumping raw base64 text
            # into the /uploads page's field list; the dedicated *_path
            # columns let that page render them as image thumbnails instead,
            # the same way it already does for the original upload.
            signature_stored_path = (
                save_bytes(signature, "signatures", f"{file.filename}.png") if signature else None
            )
            bangla_name_stored_path = (
                save_bytes(bangla_name, "bangla_names", f"{file.filename}.png") if bangla_name else None
            )
            record_upload(
                "front", file.filename, str(stored_path) if stored_path else None, ocr, True, asdict(result),
                signature_path=str(signature_stored_path) if signature_stored_path else None,
                bangla_name_path=str(bangla_name_stored_path) if bangla_name_stored_path else None,
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
        except NIDOCRError as e:
            record_upload("front", file.filename, str(stored_path) if stored_path else None, ocr, False, error_message=str(e))
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error processing {file.filename}")
            record_upload("front", file.filename, str(stored_path) if stored_path else None, ocr, False, error_message="Internal processing error.")
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
