import easyocr
from .base import OCREngine
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class EasyOCREngine(OCREngine):
    def __init__(self, languages: list[str], gpu: bool = False):
        logger.info(f"Initializing EasyOCR with languages={languages}, gpu={gpu}")
        self._reader = easyocr.Reader(languages, gpu=gpu)

    def extract(self, image_path: str) -> list[str]:
        try:
            results = self._reader.readtext(image_path)
            return [text.strip() for (_, text, _) in results if text.strip()]
        except Exception as e:
            logger.warning(f"EasyOCR failed on {image_path}: {e}")
            return []
