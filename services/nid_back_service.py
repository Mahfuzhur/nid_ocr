import shutil
import tempfile

from nid_ocr.components.preprocessing.image_preprocessor import ImagePreprocessor
from nid_ocr.components.ocr.base import OCREngine
from nid_ocr.components.detection.format_detector import NIDFormatDetector
from nid_ocr.components.extraction.base import FieldExtractor
from nid_ocr.domain.models import NIDBackData
from nid_ocr.core.logging import get_logger
from nid_ocr.services.adaptive_ocr import back_is_complete, run_adaptive_ocr

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
            fields, _ = run_adaptive_ocr(
                image_path, temp_dir, 'back', engine, self._preprocessor,
                self._detector, self._extractor, back_is_complete,
            )

            fields.pop("mrz_name", None)

            return NIDBackData(**fields)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
