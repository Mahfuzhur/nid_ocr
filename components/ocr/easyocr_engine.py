import cv2
import threading
from .base import OCREngine
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class EasyOCREngine(OCREngine):
    def __init__(self, languages: list[str], gpu: bool = False, min_confidence: float = 0.0):
        self._languages = languages
        self._gpu = gpu
        self._reader = None
        self._load_lock = threading.Lock()
        self._min_conf = min_confidence

    def _ensure_loaded(self):
        if self._reader is not None:
            return
        with self._load_lock:
            if self._reader is not None:
                return
            import easyocr

            logger.info(f"Initializing EasyOCR with languages={self._languages}, gpu={self._gpu}")
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)

    def extract(self, image_path: str) -> list[str]:
        try:
            self._ensure_loaded()
            results = self._reader.readtext(image_path)

            # Barcode filter: Smart NID back cards have a barcode in the top
            # ~15% of the image (first-row mean < 170 when barcode is present).
            # Filter out EasyOCR results whose bbox top falls in that zone so
            # barcode noise doesn't reach the extractor.
            barcode_zone = 0
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and float(img[0, :].mean()) < 170:
                    barcode_zone = img.shape[0] * 15 // 100
                    logger.info(f"EasyOCR: barcode zone detected, filtering top {barcode_zone}px")
            except Exception:
                pass

            out: list[str] = []
            for (bbox, text, conf) in results:
                text_s = text.strip()
                if not text_s or conf < self._min_conf:
                    continue
                if barcode_zone:
                    bbox_top = min(int(pt[1]) for pt in bbox)
                    if bbox_top < barcode_zone:
                        continue
                out.append(text_s)
            return out
        except Exception as exc:
            logger.warning(f"EasyOCR failed ({type(exc).__name__})")
            return []
