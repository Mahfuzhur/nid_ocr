import re
import unicodedata
from .base import Transliterator
from .digit_converter import convert_bangla_digits
from .term_dictionary import TermDictionary
from .phonetic_mapper import phonetic_map
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

_HAS_BANGLA = re.compile(r'[ঀ-৿]')


class IndicNLPTransliterator(Transliterator):
    """
    Three-stage Bengali→English transliterator (fully offline):

    1. Bangla digit conversion  (০→0 … ৯→9)
    2. Term dictionary          (চৌধুরী→Chowdhury, ঢাকা→Dhaka, etc.)
    3. Character-level phonetic mapper for any remaining Bangla chars
    """

    def __init__(self, dictionary: TermDictionary):
        self._dict = dictionary

    def transliterate(self, text: str) -> str:
        if not text:
            return text

        # Stage 1: digits
        text = convert_bangla_digits(text)

        # NFKC normalization: OCR often returns composed য় (U+09DF) while
        # dict keys are decomposed (য+়). NFKC decomposes compatibility forms
        # so they match the dict keys and the phonetic mapper pair-lookup.
        text = unicodedata.normalize('NFKC', text)

        # Stage 2: dictionary (longest-match, applied in sorted order)
        text = self._dict.apply(text)

        # Stage 3: phonetic map for remaining Bangla chars
        if _HAS_BANGLA.search(text):
            text = phonetic_map(text)

        return text.strip()
