import re
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# Smart NID markers present in OCR output
_SMART_PATTERNS = [
    re.compile(r'NID\s*No', re.IGNORECASE),
    re.compile(r'Issue\s*Date', re.IGNORECASE),
    re.compile(r'Place\s*of\s*Birth', re.IGNORECASE),
    re.compile(r'I<BGD'),          # MRZ line 1
]


class NIDFormatDetector:
    def detect(self, segments: list[str]) -> NIDFormat:
        combined = ' '.join(segments)
        for pattern in _SMART_PATTERNS:
            if pattern.search(combined):
                logger.info("NID format detected: SMART")
                return NIDFormat.SMART
        logger.info("NID format detected: OLD")
        return NIDFormat.OLD
