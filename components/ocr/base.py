from abc import ABC, abstractmethod


class OCREngine(ABC):
    prefers_original_image: bool = False

    @abstractmethod
    def extract(self, image_path: str) -> list[str]:
        """Extract text segments from a preprocessed image file."""
        ...
