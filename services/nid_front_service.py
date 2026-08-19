import shutil
import tempfile

from nid_ocr.components.preprocessing.image_preprocessor import ImagePreprocessor
from nid_ocr.components.ocr.base import OCREngine
from nid_ocr.components.detection.format_detector import NIDFormatDetector
from nid_ocr.components.extraction.base import FieldExtractor
from nid_ocr.components.signature.signature_extractor import SignatureExtractor
from nid_ocr.components.signature.bangla_name_extractor import BanglaNameExtractor
from nid_ocr.components.signature.card_analysis import analyze_front_card
from nid_ocr.domain.models import NIDFrontData
from nid_ocr.core.logging import get_logger
from nid_ocr.services.adaptive_ocr import front_is_complete, run_adaptive_ocr

logger = get_logger(__name__)


class NIDFrontService:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        engines: dict[str, OCREngine],
        detector: NIDFormatDetector,
        extractor: FieldExtractor,
        signature_extractor: SignatureExtractor,
        bangla_name_extractor: BanglaNameExtractor,
    ):
        self._preprocessor = preprocessor
        self._engines = engines
        self._detector = detector
        self._extractor = extractor
        self._signature_extractor = signature_extractor
        self._bangla_name_extractor = bangla_name_extractor

    def process(self, image_path: str, ocr: str = 'auto') -> tuple[NIDFrontData, bytes | None, bytes | None]:
        engine = self._engines.get(ocr) or self._engines['auto']
        temp_dir = tempfile.mkdtemp()
        try:
            fields, fmt = run_adaptive_ocr(
                image_path, temp_dir, 'front', engine, self._preprocessor,
                self._detector, self._extractor, front_is_complete,
            )
            analysis = analyze_front_card(image_path)
            signature = self._signature_extractor.extract_from_analysis(analysis, fmt)
            bangla_name = self._bangla_name_extractor.extract_from_analysis(analysis)

            return NIDFrontData(**fields), signature, bangla_name
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
