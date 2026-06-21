from abc import ABC, abstractmethod
from nid_ocr.domain.enums import NIDFormat


class FieldExtractor(ABC):
    @abstractmethod
    def extract(self, segments: list[str], fmt: NIDFormat) -> dict:
        """Extract structured fields from OCR segments based on the NID format."""
        ...
