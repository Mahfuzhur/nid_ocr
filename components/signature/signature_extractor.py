import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.photo_locator import locate_photo_margins
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# Fallback region (used only if no face is detected — see photo_locator.py)
# calibrated the same way as before: a fixed fraction of the canonical
# card's width/height, per format.
_FALLBACK_REGION = {
    NIDFormat.SMART: dict(x0=0.03, y0=0.80, y1=0.99, x1=0.30),
    NIDFormat.OLD:   dict(x0=0.02, y0=0.68, y1=0.98, x1=0.25),
}

# How far past the photo's bottom margin the crop is allowed to run. SMART
# cards have clear whitespace all the way to the card edge below the
# signature, so it's safe to run to the bottom. OLD cards pack "Date of
# Birth"/"ID NO" text much closer beneath the signature (confirmed by direct
# pixel measurement — as little as ~1-2% of the card's height of gap), so
# running to the card edge there picks up part of that text; a modest fixed
# height clears the signature without reaching it.
_MAX_HEIGHT_BELOW_PHOTO = {
    NIDFormat.SMART: None,   # None = run to the card's bottom edge
    NIDFormat.OLD:   0.06,
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

        margins = locate_photo_margins(card, fmt)
        if margins is not None:
            x0, x1, y0 = margins
            max_height = _MAX_HEIGHT_BELOW_PHOTO.get(fmt, _MAX_HEIGHT_BELOW_PHOTO[NIDFormat.SMART])
            y1 = h if max_height is None else min(h, int(y0 + h * max_height))
        else:
            logger.info(f"Signature extraction: no face found, using fallback region for {image_path}")
            region = _FALLBACK_REGION.get(fmt, _FALLBACK_REGION[NIDFormat.SMART])
            x0 = int(w * region["x0"])
            x1 = int(w * region["x1"])
            y0 = int(h * region["y0"])
            y1 = int(h * region["y1"])

        crop = card[y0:y1, x0:x1]
        if crop.size == 0:
            logger.warning(f"Signature extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Signature extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()
