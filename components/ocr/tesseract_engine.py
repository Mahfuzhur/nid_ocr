import pytesseract
from .base import OCREngine
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = 'ben+eng', config: str = '--psm 6'):
        self._lang = lang
        self._config = config

    def extract(self, image_path: str) -> list[str]:
        try:
            raw = pytesseract.image_to_string(image_path, lang=self._lang, config=self._config)
            return [line.strip() for line in raw.splitlines() if line.strip()]
        except Exception as exc:
            logger.warning(f"Tesseract failed ({type(exc).__name__})")
            return []
