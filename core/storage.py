import shutil
import uuid
from pathlib import Path

from nid_ocr.core.config import settings
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


def save_upload(tmp_path: str, subfolder: str, original_filename: str) -> Path | None:
    """Copy an already-saved temp upload into permanent storage. Never raises —
    a storage failure shouldn't block OCR processing of the same request."""
    try:
        dest_dir = Path(settings.upload_storage_dir) / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{uuid.uuid4().hex[:8]}_{original_filename}"
        shutil.copy2(tmp_path, dest_path)
        return dest_path
    except Exception:
        logger.exception(f"Failed to save permanent copy of {original_filename}")
        return None
