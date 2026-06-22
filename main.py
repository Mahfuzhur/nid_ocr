from fastapi import FastAPI

from nid_ocr.core.config import settings
from nid_ocr.core.logging import get_logger

from nid_ocr.components.preprocessing.image_preprocessor import ImagePreprocessor
from nid_ocr.components.ocr.easyocr_engine import EasyOCREngine
from nid_ocr.components.ocr.tesseract_engine import TesseractEngine
from nid_ocr.components.ocr.composite_engine import CompositeOCREngine
from nid_ocr.components.ocr.paddle_engine import PaddleOCREngine
from nid_ocr.components.ocr.surya_engine import SuryaOCREngine
from nid_ocr.components.detection.format_detector import NIDFormatDetector
from nid_ocr.components.extraction.front_extractor import FrontFieldExtractor
from nid_ocr.components.extraction.back_extractor import BackFieldExtractor
from nid_ocr.components.transliteration.term_dictionary import TermDictionary, COMMON_TERMS
from nid_ocr.components.transliteration.indic_transliterator import IndicNLPTransliterator

from nid_ocr.services.nid_front_service import NIDFrontService
from nid_ocr.services.nid_back_service import NIDBackService

from nid_ocr.api.routes.nid_front import NIDFrontRouter
from nid_ocr.api.routes.nid_back import NIDBackRouter

logger = get_logger(__name__)

# ── Build the object graph (Composition Root) ────────────────────────────────
logger.info("Initializing components...")

preprocessor    = ImagePreprocessor(settings)
easy_engine     = EasyOCREngine(
    languages=settings.easyocr_languages,
    gpu=settings.easyocr_gpu,
    min_confidence=settings.easyocr_min_confidence,
)
tess_engine     = TesseractEngine(lang=settings.tesseract_lang, config=settings.tesseract_config)
paddle_engine   = PaddleOCREngine(lang=settings.paddle_lang)
surya_engine    = SuryaOCREngine()
engines = {
    'auto':      CompositeOCREngine([easy_engine, tess_engine]),
    'easyocr':   easy_engine,
    'tesseract': tess_engine,
    'paddle':    paddle_engine,
    'surya':     surya_engine,
}
detector        = NIDFormatDetector()
transliterator  = IndicNLPTransliterator(TermDictionary(COMMON_TERMS))

front_extractor = FrontFieldExtractor(transliterator)
back_extractor  = BackFieldExtractor(transliterator)

front_service   = NIDFrontService(preprocessor, engines, detector, front_extractor)
back_service    = NIDBackService(preprocessor, engines, detector, back_extractor)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bangladesh NID OCR Service",
    description="Fully offline OCR service for extracting and transliterating fields from Bangladesh National ID Cards.",
    version="3.0.0",
)

app.include_router(NIDFrontRouter(front_service).router, tags=["NID"])
app.include_router(NIDBackRouter(back_service).router,   tags=["NID"])

logger.info("NID OCR Service ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nid_ocr.main:app", host="0.0.0.0", port=8000, reload=False)
