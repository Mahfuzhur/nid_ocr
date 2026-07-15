import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.signature_bounds import find_signature_bbox
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# Calibrated against the standard Bangladesh NID front-side template: the
# signature sits bottom-left, under the photo. SMART (chip) and OLD
# (laminated) cards use different physical layouts, so each format has its
# own region, expressed as a fraction (x0, y0, x1, y1) of the canonical
# (upright, cropped) card's width/height. Used to locate the right ink
# cluster via find_signature_bbox (see signature_bounds.py) rather than as
# the crop bounds directly — a fixed percentage can clip part of a signature
# that sits slightly outside it — and as a fallback if no cluster is found.
_SIGNATURE_REGION = {
    NIDFormat.SMART: (0.03, 0.80, 0.28, 0.98),
    NIDFormat.OLD:   (0.00, 0.70, 0.20, 0.82),
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
        anchor = _SIGNATURE_REGION.get(fmt, _SIGNATURE_REGION[NIDFormat.SMART])

        bbox = find_signature_bbox(card, anchor)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
        else:
            ax0, ay0, ax1, ay1 = anchor
            x0, y0, x1, y1 = int(w * ax0), int(h * ay0), int(w * ax1), int(h * ay1)

        crop = card[y0:y1, x0:x1]
        if crop.size == 0:
            logger.warning(f"Signature extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Signature extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()
