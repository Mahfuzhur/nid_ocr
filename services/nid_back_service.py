import shutil
import tempfile

from nid_ocr.components.preprocessing.image_preprocessor import ImagePreprocessor
from nid_ocr.components.ocr.base import OCREngine
from nid_ocr.components.detection.format_detector import NIDFormatDetector
from nid_ocr.components.extraction.base import FieldExtractor
from nid_ocr.domain.models import NIDBackData
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class NIDBackService:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        engines: dict[str, OCREngine],
        detector: NIDFormatDetector,
        extractor: FieldExtractor,
    ):
        self._preprocessor = preprocessor
        self._engines = engines
        self._detector = detector
        self._extractor = extractor

    def process(self, image_path: str, ocr: str = 'auto') -> NIDBackData:
        engine = self._engines.get(ocr) or self._engines['auto']
        temp_dir = tempfile.mkdtemp()
        try:
            variants = self._preprocessor.preprocess(image_path, temp_dir, mode='back')

            all_segments: list[str] = []
            seen: set[str] = set()
            paths = [image_path] if getattr(engine, 'prefers_original_image', False) else list(variants.values())
            for path in paths:
                for seg in engine.extract(path):
                    if seg not in seen:
                        seen.add(seg)
                        all_segments.append(seg)

            logger.info(f"Back OCR ({ocr}): {len(all_segments)} segments collected")

            fmt = self._detector.detect(all_segments)
            fields = self._extractor.extract(all_segments, fmt)

            fields.pop("mrz_name", None)

            return NIDBackData(**fields)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
