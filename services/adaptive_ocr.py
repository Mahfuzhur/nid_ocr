from collections.abc import Callable

from nid_ocr.components.detection.format_detector import NIDFormatDetector
from nid_ocr.components.extraction.base import FieldExtractor
from nid_ocr.components.ocr.base import OCREngine
from nid_ocr.components.preprocessing.image_preprocessor import ImagePreprocessor
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

CompletionRule = Callable[[dict], bool]


def run_adaptive_ocr(
    image_path: str,
    temp_dir: str,
    mode: str,
    engine: OCREngine,
    preprocessor: ImagePreprocessor,
    detector: NIDFormatDetector,
    extractor: FieldExtractor,
    is_complete: CompletionRule,
) -> tuple[dict, NIDFormat]:
    """Run the cheapest useful OCR pass first, adding variants only as needed."""
    all_segments: list[str] = []
    seen: set[str] = set()
    passes = 0

    def apply(path: str, label: str) -> tuple[dict, NIDFormat]:
        nonlocal passes
        passes += 1
        for segment in engine.extract(path):
            if segment not in seen:
                seen.add(segment)
                all_segments.append(segment)
        fmt = detector.detect(all_segments)
        fields = extractor.extract(all_segments, fmt)
        logger.info(
            f"{mode.title()} OCR pass {passes} ({label}): "
            f"{len(all_segments)} unique segments"
        )
        return fields, fmt

    last_fields: dict = {}
    last_format: NIDFormat | None = None

    if getattr(engine, 'prefers_original_image', False):
        last_fields, last_format = apply(image_path, 'original')
        if is_complete(last_fields):
            logger.info(f"{mode.title()} OCR completed on original image")
            return last_fields, last_format

    variants = preprocessor.preprocess(image_path, temp_dir, mode=mode)
    for label, path in variants.items():
        last_fields, last_format = apply(path, label)
        if is_complete(last_fields):
            logger.info(f"{mode.title()} OCR completed after {passes} pass(es)")
            break

    if last_format is None:
        last_format = detector.detect(all_segments)
        last_fields = extractor.extract(all_segments, last_format)
    return last_fields, last_format


def front_is_complete(fields: dict) -> bool:
    return all(fields.get(key) for key in ('name', 'date_of_birth', 'nid_number', 'mother_name')) and bool(
        fields.get('father_name') or fields.get('spouse_name')
    )


def back_is_complete(fields: dict) -> bool:
    supporting = sum(bool(fields.get(key)) for key in ('blood_group', 'issue_date', 'place_of_birth'))
    return bool(fields.get('address')) and supporting >= 2
