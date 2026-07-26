import difflib
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
# থাতা/ঘাতা: single-consonant misreads of মাতা seen from sharpened-variant OCR passes
_MOTHER_FUZZY = re.compile(r'মাত|যাতা|থাতা|ঘাতা')
_SPOUSE_FUZZY = re.compile(r'পত্নী')

# Markers used exclusively in feminine Bengali names/honorifics on Bangladeshi
# NIDs (never in a father's/husband's name) — used by the positional fallback
# below to tell "this candidate is genuinely the mother's name" apart from
# "the father's own OCR dropped out and this candidate just shifted forward
# into that slot."
_FEMININE_NAME_MARKER = re.compile(r'বেগম|খাতুন|নেছা|নেসা|^মোছাঃ|^মোসাম্মৎ|^মোসাঃ')

_DOB_PATTERN = re.compile(r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}')
_DIGIT_ONLY  = re.compile(r'\d{8,17}')

_HEADER_NOISE = re.compile(
    r'Government|Bangladesh|National|Republic|People|জাতীয়|পরিচয়|গণপ্রজাতন্ত্রী|বাংলাদেশ|সরকার',
    re.IGNORECASE,
)
_ADDRESS_NOISE = re.compile(
    r'ঠিকানা|গ্রাম|রাস্তা|বাসা|হোল্ডিং|জেলা|উপজেলা|রক্তের|প্রদান|স্বাক্ষর|মেয়াদ',
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


# How similar a candidate's transliteration must be to the person's own English
# name before it's treated as a repeat of that name (from another OCR variant)
# rather than a distinct family member. Deliberately high: father/mother often
# share the person's surname and title (e.g. "কাজী মোঃ ... রশিদ"), so a low bar
# would wrongly exclude legitimate matches.
_OWN_NAME_SIMILARITY_CUTOFF = 0.78


def _nearest_unclaimed_bangla(
    segments: list[str],
    idx: int,
    claimed: set[str],
    transliterate=None,
    own_name_en: str | None = None,
) -> str | None:
    """Return the Bengali person-name-like segment nearest (by index distance) to idx
    that hasn't already been claimed by another field.

    Merging OCR segments from multiple preprocessed variants means a label and its
    value can end up far apart in the merged list (the value may even have already
    appeared *earlier*, from a different variant's pass, than the label that names
    it) — so this scans the full list by distance rather than a small forward window.
    That also means an unlabeled repeat of a name already assigned to another field
    (garbled slightly differently by another variant's OCR pass) can be numerically
    closer to a label than the real value. `claimed` only catches exact-string
    repeats, so this also filters by transliteration similarity against both the
    person's own English name and every already-claimed Bengali value, to catch
    near-duplicates the exact-match check misses.

    On a distance tie (e.g. a neighboring field's value sits immediately before
    the label, its own value immediately after), the candidate after the label
    wins — that's how these cards are actually laid out (label, then its value),
    so preferring "after" resolves ties in the direction that's actually correct
    rather than an arbitrary index-order pick.
    """
    reference_names_en = [own_name_en] if own_name_en else []
    if transliterate:
        reference_names_en += [transliterate(v) or '' for v in claimed]

    candidates = []
    for i, s in enumerate(segments):
        if i == idx:
            continue
        c = _clean(s)
        if not c or c in claimed:
            continue
        if not _is_bangla(c) or _HEADER_NOISE.search(c) or len(c.split()) < 2:
            continue
        if transliterate and reference_names_en:
            translit = (transliterate(c) or '').lower()
            if any(
                difflib.SequenceMatcher(None, translit, ref.lower()).ratio() > _OWN_NAME_SIMILARITY_CUTOFF
                for ref in reference_names_en
            ):
                continue
        candidates.append((abs(i - idx), 0 if i > idx else 1, c))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


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
        # Bengali values already assigned to a field, so the same OCR segment
        # (or its near-duplicate from another preprocessed variant) can't also
        # be picked up for a different field by the nearest-match search below.
        claimed: set[str] = set()

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
                if name_bn:
                    claimed.add(name_bn)

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
                    father_bn = _nearest_unclaimed_bangla(
                        segments, idx, claimed, self._tr.transliterate, name_en)
                if father_bn:
                    claimed.add(father_bn)

            # Fuzzy father label (garbled stamp: '[পঙ|', 'পিত', etc.)
            elif (not father_bn and _is_short_bangla_segment(s)
                  and _FATHER_FUZZY.search(s) and not _LABEL_MOTHER.match(s)):
                father_bn = _nearest_unclaimed_bangla(
                    segments, idx, claimed, self._tr.transliterate, name_en)
                if father_bn:
                    claimed.add(father_bn)

            # ── Mother (exact label) ─────────────────────────────────────────
            m = _LABEL_MOTHER.match(s)
            if m and not mother_bn:
                inline = _clean(m.group(1))
                if inline and _is_bangla(inline):
                    mother_bn = inline
                else:
                    mother_bn = _nearest_unclaimed_bangla(
                        segments, idx, claimed, self._tr.transliterate, name_en)
                if mother_bn:
                    claimed.add(mother_bn)

            # Fuzzy mother label (garbled stamp: 'মাত', '[মাত|', etc.)
            elif (not mother_bn and _is_short_bangla_segment(s)
                  and _MOTHER_FUZZY.search(s) and not _LABEL_FATHER.match(s)):
                mother_bn = _nearest_unclaimed_bangla(
                    segments, idx, claimed, self._tr.transliterate, name_en)
                if mother_bn:
                    claimed.add(mother_bn)

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
                if spouse_bn:
                    claimed.add(spouse_bn)
            elif (not spouse_bn and _is_short_bangla_segment(s)
                  and _SPOUSE_FUZZY.search(s)):
                c = _next_non_empty(segments, idx)
                if c and _is_bangla(c):
                    spouse_bn = c
                    claimed.add(spouse_bn)

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
        # Only needed if label-based extraction missed any field. Segment order
        # doesn't reliably follow visual card layout once OCR results from
        # multiple preprocessed variants are merged (a DOB/NID line can precede
        # the name lines in the merged list even though it follows them on the
        # card), so this filters by content — excluding address-block text and
        # anything already claimed by a label match — rather than cutting the
        # list off at the first DOB/NID occurrence by position.
        if not (name_bn or name_en) or not father_bn or not mother_bn:
            bn_names = [s2.strip() for s2 in segments
                        if _looks_like_person_name_bn(s2)
                        and s2.strip() not in claimed
                        and not _ADDRESS_NOISE.search(s2)]

            # A label-less repeat of the person's own name (found only via name_en,
            # so never added to `claimed`), or of a role that's about to be filled
            # positionally below (father/mother OCR'd twice, slightly differently
            # garbled, by two preprocessed variants — the two copies sit side by
            # side in `bn_names` since neither was ever claimed by a label match),
            # would otherwise occupy an extra slot and either push later candidates
            # one position off or get wrongly assigned as e.g. spouse. Dedup by
            # transliteration similarity against every candidate already kept,
            # seeded with the person's own English name, so only the first-seen
            # copy of each distinct person survives.
            kept_translits = [name_en] if name_en else []
            deduped = []
            for c in bn_names:
                translit = (self._tr.transliterate(c) or '').lower()
                if any(
                    difflib.SequenceMatcher(None, translit, ref.lower()).ratio() > _OWN_NAME_SIMILARITY_CUTOFF
                    for ref in kept_translits
                ):
                    continue
                deduped.append(c)
                kept_translits.append(translit)
            bn_names = deduped

            # Card order is fixed — নাম, then পিতা/স্বামী, then মাতা, then
            # পত্নী — but a role's OCR can drop out entirely (not just go
            # unlabeled), which shifts every candidate after it forward by
            # one position. Filling strictly by list order would then hand
            # the next role's candidate to the wrong slot (e.g. the mother's
            # name landing in father_name). So `remaining` is consumed by
            # content where a role has a distinguishing marker (father/mother
            # via _FEMININE_NAME_MARKER), and by plain order otherwise.
            remaining = list(bn_names)

            if not name_bn and not name_en and remaining:
                name_bn = remaining.pop(0)
                logger.info(f"Positional name: {name_bn}")

            if not father_bn:
                idx = next((i for i, c in enumerate(remaining) if not _FEMININE_NAME_MARKER.search(c)), None)
                if idx is not None:
                    father_bn = remaining.pop(idx)
                    logger.info(f"Positional father: {father_bn}")

            if not mother_bn:
                idx = next((i for i, c in enumerate(remaining) if _FEMININE_NAME_MARKER.search(c)),
                           0 if remaining else None)
                if idx is not None:
                    mother_bn = remaining.pop(idx)
                    logger.info(f"Positional mother: {mother_bn}")

            if not spouse_bn and remaining:
                spouse_bn = remaining.pop(0)
                logger.info(f"Positional spouse: {spouse_bn}")

        # ── Transliterate Bengali fields ──────────────────────────────────
        return {
            "name":          name_en or (self._tr.transliterate(name_bn) if name_bn else None),
            "father_name":   self._tr.transliterate(father_bn) if father_bn else None,
            "mother_name":   self._tr.transliterate(mother_bn) if mother_bn else None,
            "spouse_name":   self._tr.transliterate(spouse_bn) if spouse_bn else None,
            "date_of_birth": date_of_birth,
            "nid_number":    nid_number,
            "name_bn":         name_bn,
            "father_name_bn":  father_bn,
            "mother_name_bn":  mother_bn,
            "spouse_name_bn":  spouse_bn,
        }
