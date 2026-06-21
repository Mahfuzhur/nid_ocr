import re
from .base import FieldExtractor
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.components.transliteration.base import Transliterator
from nid_ocr.components.transliteration.digit_converter import convert_bangla_digits
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

_LABEL_ADDRESS   = re.compile(r'ঠিকানা[:\s।]*')
# Blood group: allow digits (OCR confuses A→4, O→0)
_LABEL_BLOOD_EN  = re.compile(r'Blood\s*Group[:\s]*([A-Z0-9]{1,2}[+\-])', re.IGNORECASE)
_LABEL_BLOOD_ANY = re.compile(r'\b(AB|A|B|O|4|0)[+\-]')
# Place of birth: allow "or" as OCR misread of "of"
_LABEL_POB       = re.compile(r'Place\s+o[fr]\s*Birth[:\s]*(.+)', re.IGNORECASE)
_LABEL_ISSUE_OLD = re.compile(r'প্রদানের\s*তারিখ[:\s।]*(.+)')
_LABEL_ISSUE_NEW = re.compile(r'Issue\s*Date[:\s]*(.*)', re.IGNORECASE)
_DATE_PATTERN    = re.compile(
    r'\d{1,2}[\/\-.\s]\d{1,2}[\/\-.\s]\d{4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4}'
)
_STOP_SEGMENTS   = re.compile(
    r'(রক্তের|Blood|প্রদান|Issue|Place|signature|স্বাক্ষর|মেয়াদ)',
    re.IGNORECASE,
)
_MRZ_LINE3       = re.compile(r'^[A-Z<]{20,}$')

# OCR substitutions common in blood group values
_BG_FIXES = str.maketrans({'4': 'A', '0': 'O', '|': 'I'})


def _normalize_blood_group(raw: str) -> str:
    return raw.strip().upper().translate(_BG_FIXES)


def _normalise_date(raw: str) -> str:
    raw = convert_bangla_digits(raw.strip())
    raw = re.sub(r'[\s.]', '/', raw)
    raw = re.sub(r'-', '/', raw)
    return raw


def _parse_mrz_name(segments: list[str]) -> str | None:
    for seg in segments:
        if _MRZ_LINE3.match(seg) and '<<' in seg:
            parts = seg.split('<<')
            surname = parts[0].replace('<', ' ').strip()
            given   = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ''
            full    = f"{given} {surname}".strip()
            if full:
                return full
    return None


class BackFieldExtractor(FieldExtractor):
    def __init__(self, transliterator: Transliterator):
        self._tr = transliterator

    def extract(self, segments: list[str], fmt: NIDFormat) -> dict:
        address_parts: list[str] = []
        blood_group:   str | None = None
        issue_date:    str | None = None
        place_of_birth: str | None = None
        in_address = False

        for idx, seg in enumerate(segments):
            seg_s = seg.strip()

            # ── Address block ─────────────────────────────────────────────
            if _LABEL_ADDRESS.search(seg_s):
                # Reset on each new ঠিকানা: label so a cleaner second OCR
                # pass overwrites garbled content from the first pass.
                in_address = True
                address_parts = []
                inline = _LABEL_ADDRESS.sub('', seg_s).strip()
                if inline:
                    address_parts.append(inline)
                continue

            if in_address:
                if _STOP_SEGMENTS.search(seg_s):
                    in_address = False
                elif len(seg_s) > 2:
                    address_parts.append(seg_s)
                    continue

            # ── Blood Group ───────────────────────────────────────────────
            if not blood_group:
                m = _LABEL_BLOOD_EN.search(seg_s)
                if m:
                    blood_group = _normalize_blood_group(m.group(1))
                else:
                    m = _LABEL_BLOOD_ANY.search(seg_s)
                    if m:
                        blood_group = _normalize_blood_group(m.group(0))

            # ── Issue Date ────────────────────────────────────────────────
            if not issue_date:
                if fmt == NIDFormat.OLD:
                    m = _LABEL_ISSUE_OLD.search(seg_s)
                    if m:
                        date_raw = convert_bangla_digits(m.group(1).strip())
                        dm = _DATE_PATTERN.search(date_raw)
                        if dm:
                            issue_date = _normalise_date(dm.group(0))
                else:
                    m = _LABEL_ISSUE_NEW.search(seg_s)
                    if m:
                        inline_date = m.group(1).strip()
                        dm = _DATE_PATTERN.search(inline_date)
                        if dm:
                            issue_date = dm.group(0).strip()
                        else:
                            # Date on the next segment (label-only line)
                            for off in range(1, 3):
                                if idx + off < len(segments):
                                    nxt = segments[idx + off].strip()
                                    dm = _DATE_PATTERN.search(nxt)
                                    if dm:
                                        issue_date = dm.group(0).strip()
                                        break

            # ── Place of Birth (Smart NID only) ───────────────────────────
            if fmt == NIDFormat.SMART and not place_of_birth:
                m = _LABEL_POB.search(seg_s)
                if m:
                    val = m.group(1).strip()
                    if val:
                        place_of_birth = val

        # ── Transliterate address ─────────────────────────────────────────
        raw_address = ', '.join(address_parts) if address_parts else None
        address_en  = self._tr.transliterate(raw_address) if raw_address else None

        # ── MRZ name (Smart NID bonus) ────────────────────────────────────
        mrz_name = _parse_mrz_name(segments) if fmt == NIDFormat.SMART else None

        result: dict = {
            "address":        address_en,
            "blood_group":    blood_group,
            "issue_date":     issue_date,
            "place_of_birth": place_of_birth,
        }
        if mrz_name:
            result["mrz_name"] = mrz_name

        return result
