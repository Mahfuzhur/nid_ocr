import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# Calibrated against the standard Bangladesh NID front-side template: the
# signature sits directly below the photo, bottom-left of the card. Rather
# than guessing a tight box around the signature itself (fragile — either
# clips it or, when widened enough to search for ink, risks latching onto
# unrelated text like "ID NO"), the region simply spans the photo's left-right
# extent and runs from the photo's bottom edge down to the bottom of the
# card. SMART (chip) and OLD (laminated) cards use different physical
# layouts, so each format has its own (x0, y0, x1, y1) as a fraction of the
# canonical (upright, cropped) card's width/height.
_SIGNATURE_REGION = {
    NIDFormat.SMART: (0.03, 0.80, 0.30, 0.99),
    NIDFormat.OLD:   (0.00, 0.68, 0.20, 0.98),
}


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
        x0, y0, x1, y1 = _SIGNATURE_REGION.get(fmt, _SIGNATURE_REGION[NIDFormat.SMART])
        crop = card[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        if crop.size == 0:
            logger.warning(f"Signature extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Signature extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()
