import re
from .base import FieldExtractor
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.components.transliteration.base import Transliterator
from nid_ocr.components.transliteration.digit_converter import convert_bangla_digits
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

_LABEL_ADDRESS   = re.compile(r'ঠিকানা[:\s।]*')
# House/Holding: far more OCR-reliable than ঠিকানা itself — ঠিকানা often
# loses its leading characters, but this label (present in Bangla + English
# on every NID back) tends to survive. Used as the true start-of-address
# anchor so any boilerplate ("this card is government property...") that OCR
# merges in ahead of it is dropped rather than swept into the address.
_LABEL_HOUSE     = re.compile(
    r'(?:বাসা|বাড়ি)\s*/?\s*(?:হোল্ডিং|হোভিং|হোতিং)|House\s*/\s*Holding',
    re.IGNORECASE,
)
# গ্রাম/রাস্তা (Village/Road): another label OCR reliably recognizes, used the
# same way as _LABEL_HOUSE — as a restart anchor when it shows up mid-segment
# behind unrecognized prefix text.
_LABEL_VILLAGE   = re.compile(r'গ্রাম\s*/?\s*(?:রাস্তা|রন্তা|রাজ্ঞা)', re.IGNORECASE)
_HAS_BANGLA      = re.compile(r'[ঀ-৿]')
# Blood group: allow digits (OCR confuses A→4, O→0)
_LABEL_BLOOD_EN  = re.compile(r'Blood\s*Group[:\s]*([A-Z0-9]{1,2}[+\-])', re.IGNORECASE)
_LABEL_BLOOD_ANY = re.compile(r'\b(AB|A|B|O|4|0)[+\-]')
# Place of birth: allow "or" as OCR misread of "of"
_LABEL_POB       = re.compile(r'Place\s+o[fr]\s*Birth[:\s]*(.+)', re.IGNORECASE)
_LABEL_ISSUE_OLD = re.compile(r'প্রদানের\s*তারিখ[:\s।]*(.+)')
# "Issue" is frequently dropped/garbled by OCR (e.g. "e-Date:"), so only
# require "Date" — on the back of a SMART NID this label is unambiguous.
_LABEL_ISSUE_NEW = re.compile(r'(?:Issue\s*)?Date[:\s]*(.*)', re.IGNORECASE)
_DATE_PATTERN    = re.compile(
    r'\d{1,2}[\/\-.\s]\d{1,2}[\/\-.\s]\d{4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4}'
)
_STOP_SEGMENTS   = re.compile(
    r'(রক্তের|Blood|প্রদান|Issue|Place|signature|স্বাক্ষর|মেয়াদ)',
    re.IGNORECASE,
)
_MRZ_LINE3       = re.compile(r'^[A-Z<]{20,}$')
# Generic MRZ line (all 3 lines: letters/digits/'<' only) — used to end the
# address block even when a line doesn't match the stricter _MRZ_LINE3 shape.
_MRZ_ANY         = re.compile(r'^[A-Z0-9<]{15,}$')
_SPACE_SLASH     = re.compile(r'\s*/\s*')

# OCR substitutions common in blood group values
_BG_FIXES = str.maketrans({'4': 'A', '0': 'O', '|': 'I'})

# A known label immediately followed by ';' — OCR sometimes puts a semicolon
# where the label's own colon belongs (e.g. "গ্রাম/রাস্তা;"). Must be fixed to
# ':' *before* the generic ';'->',' rule in _clean_address_text, or the label
# loses its colon and gets merged into the surrounding address text instead.
_LABEL_SEMICOLON = re.compile(
    rf'((?:{_LABEL_HOUSE.pattern})|(?:{_LABEL_VILLAGE.pattern}))\s*;',
    re.IGNORECASE,
)


def _normalize_blood_group(raw: str) -> str:
    return raw.strip().upper().translate(_BG_FIXES)


def _slice_from_house(seg_s: str, house_m: re.Match) -> str:
    # Anchor on the House/Holding match, but ensure it keeps a colon even when
    # OCR dropped it entirely (label runs straight into the value with just a
    # space, e.g. "বাসা/হোতিং ঠাকুর বাড়ি...").
    label = seg_s[house_m.start():house_m.end()]
    rest = seg_s[house_m.end():].lstrip()
    if not rest.startswith(':'):
        rest = f': {rest}' if rest else ':'
    return f'{label}{rest}'.strip()


def _is_address_stop(seg_s: str) -> bool:
    # Beyond the literal Blood/Issue/etc. keywords (which OCR often mangles —
    # "Blood Group" -> "Clinor Group", "Issue Date" -> "e-Date"), also stop on
    # the *shape* of a blood-group value, a full date, or an MRZ line, since
    # none of those ever legitimately appear inside an address.
    return bool(
        _STOP_SEGMENTS.search(seg_s)
        or _LABEL_BLOOD_ANY.search(seg_s)
        or _DATE_PATTERN.search(seg_s)
        or _MRZ_ANY.match(seg_s.replace(' ', ''))
    )


def _clean_address_text(raw: str) -> str:
    # Strip Devanagari block (U+0900–U+097F): Surya occasionally mistakes
    # Bengali script for Hindi, producing Devanagari garbage characters.
    raw = re.sub(r'[ऀ-ॿ]+', '', raw)
    raw = re.sub(r'\s+', ' ', raw)
    raw = _SPACE_SLASH.sub('/', raw)        # "বাসা /হোল্ডিং" → "বাসা/হোল্ডিং"
    raw = _LABEL_SEMICOLON.sub(lambda m: m.group(1) + ':', raw)  # label's own ';' -> ':'
    raw = raw.replace(';', ',')             # any other OCR misread of ',' as ';'
    raw = re.sub(r',\s*,', ',', raw)        # collapse consecutive commas
    # Segment joins add a comma before "- <pincode>" but it should be a space-dash.
    # e.g. "Para Dagair, - 1216" → "Para Dagair - 1216"
    raw = re.sub(r',\s+(-\s*\d)', r' \1', raw)
    return raw.strip(', -')


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
            addr_m = _LABEL_ADDRESS.search(seg_s)
            house_m = _LABEL_HOUSE.search(seg_s)
            if addr_m or house_m:
                # Reset on each new label so a cleaner second OCR pass
                # overwrites garbled content from the first pass.
                # Slice from the match's start/end (not .sub()) so any
                # boilerplate text OCR merged onto the same line *before* the
                # label — e.g. the "card is government property" notice — is
                # dropped along with it, rather than kept as an address prefix.
                # House/Holding wins when both are found: it survives OCR far
                # more reliably than ঠিকানা, so it's the truer anchor.
                in_address = True
                address_parts = []
                inline = _slice_from_house(seg_s, house_m) if house_m else seg_s[addr_m.end():].strip()
                if inline:
                    address_parts.append(inline)
                continue

            if in_address:
                if _is_address_stop(seg_s):
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
                    raw_val = m.group(1).strip()
                    # Capture only the uppercase place name; stop at first lowercase
                    # word (garbled OCR noise from adjacent lines merged by OCR).
                    uc_m = re.match(r'^([A-Z][A-Z\s\-]+?)(?:\s+[a-z]|$)', raw_val)
                    val = uc_m.group(1).strip() if uc_m else raw_val.split()[0] if raw_val else ''
                    if val:
                        place_of_birth = val

        # ── Positional address fallback ───────────────────────────────────
        # When ঠিকানা: label was garbled by OCR, collect all Bengali-bearing
        # segments that appear before the first stop trigger (Blood/Issue/Place).
        if not address_parts:
            stop_idx = len(segments)
            for i, s in enumerate(segments):
                if _is_address_stop(s.strip()):
                    stop_idx = i
                    break
            candidates = segments[:stop_idx]
            # Prefer restarting from the House/Holding anchor: it survives OCR
            # far more reliably than ঠিকানা, so it catches cases (e.g. the
            # notice text sitting on its own lines before the address) where
            # ঠিকানা was garbled beyond recognition and would otherwise let the
            # boilerplate get swept in as address content.
            restarted = False
            for i, s in enumerate(candidates):
                house_m = _LABEL_HOUSE.search(s.strip())
                if house_m:
                    first = _slice_from_house(s.strip(), house_m)
                    candidates = ([first] if first else []) + list(candidates[i + 1:])
                    restarted = True
                    break
            # If the ঠিকানা: label survived somewhere in this range but the
            # main loop above missed it, start from there so the "card is
            # government property" notice text isn't swept in as address.
            if not restarted:
                for i, s in enumerate(candidates):
                    label_m = _LABEL_ADDRESS.search(s.strip())
                    if label_m:
                        first = s.strip()[label_m.end():].strip()
                        candidates = ([first] if first else []) + list(candidates[i + 1:])
                        break
            for s in candidates:
                s_s = s.strip()
                if len(s_s) > 3 and _HAS_BANGLA.search(s_s):
                    address_parts.append(s_s)
            if address_parts:
                logger.info(f"Address positional fallback: {len(address_parts)} segments before stop idx {stop_idx}")

        # ── Transliterate address ─────────────────────────────────────────
        raw_address = _clean_address_text(', '.join(address_parts)) if address_parts else None
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
