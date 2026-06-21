"""
Bengali (Bangla) to Latin phonetic mapper.

Applies standard Bangladeshi romanization rules:
- Each consonant carries an inherent 'a' vowel unless suppressed by a
  matra or virama (্)
- Trailing inherent 'a' is dropped at word end
- Each Bengali word is title-cased independently
- Non-Bengali characters (ASCII, digits, punctuation) pass through unchanged
"""
import unicodedata
import re

_CONSONANTS: dict[str, str] = {
    'ক': 'k',  'খ': 'kh', 'গ': 'g',  'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh','জ': 'j',  'ঝ': 'jh', 'ঞ': 'n',
    'ট': 't',  'ঠ': 'th', 'ড': 'd',  'ঢ': 'dh', 'ণ': 'n',
    'ত': 't',  'থ': 'th', 'দ': 'd',  'ধ': 'dh', 'ন': 'n',
    'প': 'p',  'ফ': 'f',  'ব': 'b',  'ভ': 'bh', 'ম': 'm',
    'য': 'j',  'র': 'r',  'ল': 'l',  'শ': 'sh', 'ষ': 'sh',
    'স': 's',  'হ': 'h',
    'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y', 'ৎ': 't',
}

_IND_VOWELS: dict[str, str] = {
    'অ': 'o', 'আ': 'a', 'ই': 'i', 'ঈ': 'i',
    'উ': 'u', 'ঊ': 'u', 'ঋ': 'ri',
    'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
}

_MATRAS: dict[str, str] = {
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u',
    'ৃ': 'ri', 'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou',
}

_SPECIAL: dict[str, str] = {
    'ং': 'ng', 'ঃ': 'h', 'ঁ': 'n',
}

_VIRAMA = '্'
_NUKTA  = '়'  # combining nukta — U+09BC

# Decomposed nukta consonant pairs (after NFKC, ড়/ঢ়/য় appear as base+nukta)
_NUKTA_PAIRS: dict[str, str] = {
    'য' + _NUKTA: 'y',   # য + ় = য়
    'ড' + _NUKTA: 'r',   # ড + ় = ড়
    'ঢ' + _NUKTA: 'rh',  # ঢ + ় = ঢ়
}

# ৎ (khanda ta) never carries an inherent vowel.
# When followed by another consonant it acts as a cluster-initial half-consonant,
# and the following consonant gets the inherent 'a' even at word-end.
_KHANDA_TA = 'ৎ'

_BENGALI_RE = re.compile(r'[ঀ-৿]+')


def _map_word(word: str) -> str:
    """Map a single Bengali word string to title-cased Latin phonetics."""
    chars = list(word)
    result: list[str] = []
    n = len(chars)
    i = 0
    prev_was_khanda = False   # True after ৎ so next consonant forces inherent 'a'

    while i < n:
        c = chars[i]

        # ── Decomposed nukta pair (ড+়, ঢ+়, য+়) ──────────────────────────
        if i + 1 < n and chars[i + 1] == _NUKTA:
            pair = c + _NUKTA
            if pair in _NUKTA_PAIRS:
                result.append(_NUKTA_PAIRS[pair])
                i += 2  # consume base + nukta
                # Inherent vowel logic for the pair
                if i < n:
                    nxt = chars[i]
                    if nxt == _VIRAMA:
                        i += 1; prev_was_khanda = False; continue
                    elif nxt in _MATRAS:
                        result.append(_MATRAS[nxt]); i += 1
                        prev_was_khanda = False; continue
                    else:
                        result.append('a')
                elif prev_was_khanda:
                    result.append('a')
                prev_was_khanda = False
                continue

        # ── ৎ (khanda ta): no inherent vowel; next consonant gets forced 'a' ──
        if c == _KHANDA_TA:
            result.append('t')
            prev_was_khanda = (i + 1 < n)  # True only if more chars follow
            i += 1
            continue

        if c in _CONSONANTS:
            result.append(_CONSONANTS[c])
            if i + 1 < n:
                nxt = chars[i + 1]
                if nxt == _VIRAMA:
                    i += 2; prev_was_khanda = False; continue
                elif nxt in _MATRAS:
                    result.append(_MATRAS[nxt]); i += 2
                    prev_was_khanda = False; continue
                else:
                    result.append('a')  # inherent vowel before another char
            elif prev_was_khanda:
                result.append('a')   # last consonant after ৎ gets inherent 'a'
            prev_was_khanda = False

        elif c in _IND_VOWELS:
            result.append(_IND_VOWELS[c])
            prev_was_khanda = False

        elif c in _MATRAS:
            result.append(_MATRAS[c])  # orphaned matra
            prev_was_khanda = False

        elif c in (_VIRAMA, _NUKTA):
            pass  # skip combining marks not consumed above

        elif c in _SPECIAL:
            result.append(_SPECIAL[c])
            prev_was_khanda = False

        else:
            result.append(c)
            prev_was_khanda = False

        i += 1

    phonetic = ''.join(result)
    # Title-case: first letter upper, rest lower
    return phonetic[0].upper() + phonetic[1:].lower() if phonetic else ''


def phonetic_map(text: str) -> str:
    """
    Transliterate Bengali Unicode spans within text to phonetic Latin.
    Non-Bengali segments pass through unchanged, preserving their original case.
    """
    if not text:
        return text

    # NFKC pre-composes nukta sequences: য+় → য়, ড+় → ড়, etc.
    text = unicodedata.normalize('NFKC', text)

    result: list[str] = []
    prev_end = 0

    for m in _BENGALI_RE.finditer(text):
        # Append non-Bengali segment before this match unchanged
        if m.start() > prev_end:
            result.append(text[prev_end:m.start()])
        # Map Bengali word to phonetic Latin and title-case
        result.append(_map_word(m.group()))
        prev_end = m.end()

    # Append any trailing non-Bengali segment
    if prev_end < len(text):
        result.append(text[prev_end:])

    return ''.join(result)
