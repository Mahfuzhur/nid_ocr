import re
from .base import FieldExtractor
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.components.transliteration.base import Transliterator
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

_BN = re.compile(r'[ঀ-৿]')

# Label matchers — exact
# Covers OCR variants: "Name", "Name:", "Narne" (stamp garble m→rn), "Nane"
_LABEL_NAME_EN = re.compile(r'^(?:Name?|Narne|Nane?)\s*[:\.]?\s*(.*)$', re.IGNORECASE)
_LABEL_NAME_BN = re.compile(r'^নাম[:\s।]*(.*)$')
# স্বামী (husband) fills the guardian slot on female NIDs — maps to father_name
# পিতা/স্বামী is the combined label on older cards; Surya sometimes reads only স্বামী
_LABEL_FATHER  = re.compile(r'^(?:পিতা|পিতাঃ|পিতা:|পিতা/স্বামী|স্বামী|স্বামী:)[:\s।]*(.*)$')
# যাতা is Surya's common misread of মাতা (ম→য)
_LABEL_MOTHER  = re.compile(r'^(?:মাতা|মাতাঃ|মাতা:|যাতা|যাতা:)[:\s।]*(.*)$')
# পত্নী (wife) only — স্বামী now handled by father label above
_LABEL_SPOUSE  = re.compile(r'^(?:পত্নী|পত্নী:)[:\s।]*(.*)$')
_LABEL_DOB     = re.compile(r'Date\s+of\s+Birth[:\s]*(.+)', re.IGNORECASE)
_LABEL_NID_OLD = re.compile(r'ID\s*NO[:\s]*(.+)', re.IGNORECASE)
_LABEL_NID_NEW = re.compile(r'NID\s*No[\.;]?[:\s]*(.+)', re.IGNORECASE)

# Fuzzy label matchers — handle stamp/noise garbling (prefix match + short segment)
_FATHER_FUZZY = re.compile(r'পিত|স্বামী')
_MOTHER_FUZZY = re.compile(r'মাত|যাতা')
_SPOUSE_FUZZY = re.compile(r'পত্নী')

_DOB_PATTERN = re.compile(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}')
_DIGIT_ONLY  = re.compile(r'\d{8,17}')

_HEADER_NOISE = re.compile(
    r'Government|Bangladesh|National|Republic|People|জাতীয়|পরিচয়|গণপ্রজাতন্ত্রী|বাংলাদেশ|সরকার',
    re.IGNORECASE,
)


def _is_bangla(text: str) -> bool:
    return bool(_BN.search(text))


def _clean(text: str) -> str | None:
    text = re.sub(r'^[:\s।\-—|]+|[:\s।\-—|]+$', '', text)
    return text if len(text) > 1 else None


def _next_non_empty(segments: list[str], idx: int, window: int = 3) -> str | None:
    for i in range(idx + 1, min(idx + window + 1, len(segments))):
        val = _clean(segments[i])
        if val and not _HEADER_NOISE.search(val):
            return val
    return None


def _next_bangla(segments: list[str], idx: int, window: int = 4) -> str | None:
    """Return the first segment within window that has ≥2 Bengali words."""
    for i in range(idx + 1, min(idx + window + 1, len(segments))):
        c = _clean(segments[i])
        if c and _is_bangla(c) and len(c.split()) >= 2:
            return c
    return None


_NAME_JUNK = re.compile(r'\b(?:ID|Card|National|Republic|Bangladesh|Government)\b', re.IGNORECASE)

def _is_clean_english_name(text: str) -> bool:
    """True if text looks like a printable English name (not garbled or noise)."""
    if not text or _is_bangla(text):
        return False
    text = text.strip()
    if _NAME_JUNK.search(text):
        return False
    # Long strings with no word separators at all are garbled (e.g. 'HDLIFTEKAARALAMUTSA').
    # Hyphens are valid in names like 'ABDULLAH-HEL-AZMAIN', so allow them as separators.
    if len(text) > 15 and not re.search(r'[\s\-]', text):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z\s\.\-']{1,}$", text))


def _looks_like_person_name_bn(seg: str) -> bool:
    """True if segment looks like a Bengali person name: ≥2 words, no header text."""
    seg = seg.strip()
    if not _is_bangla(seg):
        return False
    if _HEADER_NOISE.search(seg):
        return False
    return len(seg.split()) >= 2


def _is_short_bangla_segment(seg: str) -> bool:
    """True if segment is short Bengali text (likely a garbled label, not a name)."""
    return _is_bangla(seg) and len(seg.strip()) < 15


class FrontFieldExtractor(FieldExtractor):
    def __init__(self, transliterator: Transliterator):
        self._tr = transliterator

    def extract(self, segments: list[str], fmt: NIDFormat) -> dict:
        name_en       = None
        name_bn       = None
        father_bn     = None
        mother_bn     = None
        spouse_bn     = None
        date_of_birth = None
        nid_number    = None

        for idx, seg in enumerate(segments):
            s = seg.strip()

            # ── Bengali name (explicit label) ────────────────────────────────
            m = _LABEL_NAME_BN.match(s)
            if m and not name_bn:
                inline = _clean(m.group(1))
                if inline and _is_bangla(inline):
                    name_bn = inline
                else:
                    c = _next_non_empty(segments, idx)
                    if c and _is_bangla(c):
                        name_bn = c

            # ── English name (explicit label) ────────────────────────────────
            m = _LABEL_NAME_EN.match(s)
            if m and not name_en:
                inline = _clean(m.group(1))
                if inline and _is_clean_english_name(inline):
                    name_en = inline
                else:
                    for off in range(1, 4):
                        if idx + off < len(segments):
                            c = _clean(segments[idx + off])
                            if c and _is_clean_english_name(c):
                                name_en = c
                                break

            # ── Father (exact label) ─────────────────────────────────────────
            m = _LABEL_FATHER.match(s)
            if m and not father_bn:
                inline = _clean(m.group(1))
                if inline and _is_bangla(inline):
                    father_bn = inline
                else:
                    father_bn = _next_bangla(segments, idx)

            # Fuzzy father label (garbled stamp: '[পঙ|', 'পিত', etc.)
            elif (not father_bn and _is_short_bangla_segment(s)
                  and _FATHER_FUZZY.search(s) and not _LABEL_MOTHER.match(s)):
                father_bn = _next_bangla(segments, idx)

            # ── Mother (exact label) ─────────────────────────────────────────
            m = _LABEL_MOTHER.match(s)
            if m and not mother_bn:
                inline = _clean(m.group(1))
                if inline and _is_bangla(inline):
                    mother_bn = inline
                else:
                    mother_bn = _next_bangla(segments, idx)

            # Fuzzy mother label (garbled stamp: 'মাত', '[মাত|', etc.)
            elif (not mother_bn and _is_short_bangla_segment(s)
                  and _MOTHER_FUZZY.search(s) and not _LABEL_FATHER.match(s)):
                mother_bn = _next_bangla(segments, idx)

            # ── Spouse ──────────────────────────────────────────────────────
            m = _LABEL_SPOUSE.match(s)
            if m and not spouse_bn:
                inline = _clean(m.group(1))
                if inline and _is_bangla(inline):
                    spouse_bn = inline
                else:
                    c = _next_non_empty(segments, idx)
                    if c and _is_bangla(c):
                        spouse_bn = c
            elif (not spouse_bn and _is_short_bangla_segment(s)
                  and _SPOUSE_FUZZY.search(s)):
                c = _next_non_empty(segments, idx)
                if c and _is_bangla(c):
                    spouse_bn = c

            # ── Date of Birth ─────────────────────────────────────────────────
            if not date_of_birth:
                m = _LABEL_DOB.search(s)
                if m:
                    dob_match = _DOB_PATTERN.search(m.group(1))
                    if dob_match:
                        date_of_birth = dob_match.group(0)
                else:
                    dob_match = _DOB_PATTERN.search(s)
                    if dob_match:
                        date_of_birth = dob_match.group(0)

            # ── NID Number ───────────────────────────────────────────────────
            if not nid_number:
                label_re = _LABEL_NID_NEW if fmt == NIDFormat.SMART else _LABEL_NID_OLD
                m = label_re.search(s)
                if m:
                    raw = re.sub(r'[\s\-]', '', m.group(1))
                    digits = _DIGIT_ONLY.search(raw)
                    if digits:
                        nid_number = digits.group(0)
                    else:
                        # Label-only segment — value is on the next line
                        for off in range(1, 3):
                            if idx + off < len(segments):
                                raw2 = re.sub(r'[\s\-]', '', segments[idx + off])
                                d = _DIGIT_ONLY.search(raw2)
                                if d:
                                    nid_number = d.group(0)
                                    break

        # ── Fallback: NID digit scan ─────────────────────────────────────
        if not nid_number:
            for seg in segments:
                raw = re.sub(r'[\s\-]', '', seg)
                for d in re.findall(r'\d{10}|\d{13}|\d{17}', raw):
                    nid_number = d
                    logger.info(f"NID number found via fallback scan: {nid_number}")
                    break
                if nid_number:
                    break

        # ── Positional fallback: assign Bengali names in appearance order ──
        # Only needed if label-based extraction missed any field.
        # Limit search to segments before the first DOB/NID line so address
        # text on combined images (or noisy trailer segments) doesn't pollute.
        if not (name_bn or name_en) or not father_bn or not mother_bn:
            cutoff = len(segments)
            for i, s2 in enumerate(segments):
                if (_LABEL_DOB.search(s2) or _LABEL_NID_OLD.search(s2)
                        or _LABEL_NID_NEW.search(s2)):
                    cutoff = i
                    break

            bn_names = [s2.strip() for s2 in segments[:cutoff]
                        if _looks_like_person_name_bn(s2)]

            if bn_names and not name_bn and not name_en:
                name_bn = bn_names[0]
                logger.info(f"Positional name (bn): {name_bn}")
            if len(bn_names) > 1 and not father_bn:
                father_bn = bn_names[1]
                logger.info(f"Positional father: {father_bn}")
            if len(bn_names) > 2 and not mother_bn:
                mother_bn = bn_names[2]
                logger.info(f"Positional mother: {mother_bn}")
            if len(bn_names) > 3 and not spouse_bn:
                spouse_bn = bn_names[3]
                logger.info(f"Positional spouse: {spouse_bn}")

        # ── Transliterate Bengali fields ──────────────────────────────────
        return {
            "name":          name_en or (self._tr.transliterate(name_bn) if name_bn else None),
            "father_name":   self._tr.transliterate(father_bn) if father_bn else None,
            "mother_name":   self._tr.transliterate(mother_bn) if mother_bn else None,
            "spouse_name":   self._tr.transliterate(spouse_bn) if spouse_bn else None,
            "date_of_birth": date_of_birth,
            "nid_number":    nid_number,
        }
