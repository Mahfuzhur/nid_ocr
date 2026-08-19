import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_DISABLE_MKLDNN'] = '1'

from .base import OCREngine
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class PaddleOCREngine(OCREngine):
    def __init__(self, lang: str = 'en'):
        self._lang = lang
        self._ocr = None

    def _ensure_loaded(self):
        if self._ocr is not None:
            return
        logger.info(f"Initializing PaddleOCR lang={self._lang} on first use...")
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_textline_orientation=True, lang=self._lang)

    def extract(self, image_path: str) -> list[str]:
        try:
            self._ensure_loaded()
            results = self._ocr.predict(image_path)
            segments: list[str] = []
            for item in (results or []):
                for text in (item.get('rec_texts') or []):
                    t = text.strip() if text else ''
                    if t:
                        segments.append(t)
            return segments
        except Exception as exc:
            logger.warning(f"PaddleOCR failed ({type(exc).__name__})")
            return []
