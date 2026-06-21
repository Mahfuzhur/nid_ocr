from .base import OCREngine
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class CompositeOCREngine(OCREngine):
    """Runs multiple OCR engines and merges their results, deduplicating segments."""

    def __init__(self, engines: list[OCREngine]):
        self._engines = engines

    def extract(self, image_path: str) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for engine in self._engines:
            for segment in engine.extract(image_path):
                if segment not in seen:
                    seen.add(segment)
                    merged.append(segment)
        logger.info(f"CompositeOCR extracted {len(merged)} unique segments from {image_path}")
        return merged
