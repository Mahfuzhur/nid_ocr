import cv2
import numpy as np

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# Calibrated against the standard Bangladesh NID front-side template: the
# signature sits directly below the photo, bottom-left of the card. x0/y0/y1
# anchor the region to the photo's left edge and span from the photo's
# bottom down to the card's bottom edge (see _find_right_edge for why x1
# isn't fixed too). SMART (chip) and OLD (laminated) cards use different
# physical layouts, so each format has its own anchor, expressed as a
# fraction of the canonical (upright, cropped) card's width/height.
# x1_search is a generous upper bound for how far right a signature could
# plausibly extend — wide enough to hold a large signature, but short of the
# card's date/ID-number column so that column never has to be excluded by
# the gap search below (it's a safety bound, not an expected result).
_SIGNATURE_REGION = {
    NIDFormat.SMART: dict(x0=0.03, y0=0.80, y1=0.99, x1_search=0.60),
    NIDFormat.OLD:   dict(x0=0.02, y0=0.68, y1=0.98, x1_search=0.50),
}

_INK_THRESHOLD = 1
_MIN_START_RUN = 4      # consecutive ink columns needed to count as real signature start,
                         # not a 1px card-border artifact right at the left edge
_GAP_MIN_COLUMNS = 15
_GAP_FRACTION = 0.03     # gap also scales with band width, whichever is larger
_PADDING = 8


def _find_right_edge(col_profile: np.ndarray, band_width: int) -> int:
    """Return the column where the signature's ink ends: the start of the
    first real whitespace gap after the signature begins. This lets the crop
    grow to whatever width the actual signature needs — rather than a fixed
    x1 that clips a wide signature or, if widened generically, risks
    including the next text block — since it stops at the first genuine gap
    regardless of exactly where that falls.
    """
    run = 0
    started_at = None
    for i, v in enumerate(col_profile):
        if v > _INK_THRESHOLD:
            run += 1
            if run >= _MIN_START_RUN and started_at is None:
                started_at = i - run + 1
        else:
            run = 0

    if started_at is None:
        return band_width  # no ink found in the search band at all

    gap_needed = max(_GAP_MIN_COLUMNS, int(_GAP_FRACTION * band_width))
    gap_run = 0
    for i in range(started_at, len(col_profile)):
        if col_profile[i] > _INK_THRESHOLD:
            gap_run = 0
        else:
            gap_run += 1
            if gap_run >= gap_needed:
                return i - gap_run + 1
    return band_width


class SignatureExtractor:
    def extract(self, image_path: str, fmt: NIDFormat) -> bytes | None:
        """Return the cropped signature region as PNG bytes, or None if the
        image couldn't be read or the crop came out empty."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Signature extraction: could not read {image_path}")
            return None

        card = locate_card(img)
        h, w = card.shape[:2]
        region = _SIGNATURE_REGION.get(fmt, _SIGNATURE_REGION[NIDFormat.SMART])
        x0 = int(w * region["x0"])
        y0 = int(h * region["y0"])
        y1 = int(h * region["y1"])
        x1_search = int(w * region["x1_search"])

        band = card[y0:y1, x0:x1_search]
        if band.size == 0:
            logger.warning(f"Signature extraction: empty search band for {image_path}")
            return None

        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        col_profile = th.sum(axis=0) / 255

        right_edge = _find_right_edge(col_profile, band.shape[1])
        crop_x1 = min(band.shape[1], right_edge + _PADDING)
        crop = band[:, 0:crop_x1]
        if crop.size == 0:
            logger.warning(f"Signature extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Signature extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()
