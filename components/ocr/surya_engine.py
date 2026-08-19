import os
import re
import threading
from PIL import Image

from .base import OCREngine
from nid_ocr.core.logging import get_logger

# Bengali dependent vowel markers (U+09BE – U+09CC) and virama (U+09CD).
# When <br> is immediately followed by one of these, the line broke mid-word
# inside a Bengali syllable — join without a space so the vowel stays attached
# to its consonant rather than becoming an orphaned fragment.
_BR_BEFORE_VOWEL = re.compile(r'<br>(?=[া-ৌ্])')

logger = get_logger(__name__)


class SuryaOCREngine(OCREngine):
    prefers_original_image: bool = True

    def __init__(self):
        self._det = None
        self._rec = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._det is not None:
            return
        with self._load_lock:
            if self._det is not None:
                return
            from nid_ocr.core.config import settings

            os.environ.setdefault('RECOGNITION_BATCH_SIZE', str(settings.surya_recognition_batch_size))
            os.environ.setdefault('DETECTOR_BATCH_SIZE', str(settings.surya_detector_batch_size))

            import torch
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.settings import settings as surya_settings

            cuda_available = torch.cuda.is_available()
            use_fp16 = settings.surya_dtype == 'float16' and cuda_available
            dtype = torch.float16 if use_fp16 else torch.float32
            device = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
            logger.info(f"Initializing Surya OCR (device={device}, dtype={dtype}, takes 1-2 min)...")
            self._det = DetectionPredictor()
            foundation = FoundationPredictor(
                checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                dtype=dtype,
            )
            self._rec = RecognitionPredictor(foundation)
            logger.info("Surya OCR models loaded.")

    def warmup(self) -> None:
        self._ensure_loaded()

    def extract(self, image_path: str) -> list[str]:
        try:
            self._ensure_loaded()
            image = Image.open(image_path).convert('RGB')
            results = self._rec([image], det_predictor=self._det)
            segments: list[str] = []
            for page in results:
                for line in page.text_lines:
                    t = line.text or ''
                    t = _BR_BEFORE_VOWEL.sub('', t)      # join mid-word line breaks
                    t = t.replace('<br>', ' ')
                    t = re.sub(r'<[^>]+>', '', t)        # strip <mark>, <b>, etc.
                    t = t.strip()
                    if t:
                        segments.append(t)
            return segments
        except Exception as e:
            logger.warning(f"Surya OCR failed on {image_path}: {e}")
            return []
