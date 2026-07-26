import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.photo_locator import detect_largest_face
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# SMART (chip) cards: the Bengali "নাম:" line sits in the text column to the
# right of the photo, one line *above* the top of the detected face box (the
# face box brackets roughly the "Name:"-through-"মাতা:" rows, not নাম) —
# calibrated against a real card where the face box sat at y=43.3%-65.3% of
# card height and the নাম row at ~35%-42%. The gap fraction matches
# SignatureExtractor._SMART_MARGINS["right"], since it marks the same
# photo/text-column boundary viewed from the other side.
_SMART_RIGHT_GAP = 0.12     # of face width
_SMART_ABOVE_FACE = 0.40    # of face height, how far above the face top the row starts
_SMART_ROW_HEIGHT = 0.34    # of face height

# OLD (laminated) cards: the Bengali name sits above the photo entirely
# (right of the small header logo, before the photo/English-name block even
# starts), so it's anchored to the card itself rather than the face box —
# same reasoning as SignatureExtractor's OLD-format fixed sizing.
_OLD_BOUNDS = dict(x0=0.30, x1=0.97, y0=0.08, y1=0.20)

# Fallback region if no face is detected on a SMART card.
_SMART_NO_FACE_FALLBACK = dict(x0=0.32, x1=0.97, y0=0.20, y1=0.34)


class BanglaNameExtractor:
    def extract(self, image_path: str, fmt: NIDFormat) -> bytes | None:
        """Return the cropped Bengali name-line region as PNG bytes, or None
        if the image couldn't be read or the crop came out empty.

        These regions are a first-pass estimate from two reference card
        layouts, not content-verified against a large sample — expect to
        tune the fractions above against real uploads, the same way
        SignatureExtractor's crop went through several rounds of fixes.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Bangla name extraction: could not read {image_path}")
            return None

        card = locate_card(img)
        h, w = card.shape[:2]

        if fmt == NIDFormat.OLD:
            x0, x1, y0, y1 = self._fraction_bounds(_OLD_BOUNDS, w, h)
        else:
            face = detect_largest_face(card)
            if face is None:
                logger.info(f"Bangla name extraction: no face found, using fallback region for {image_path}")
                x0, x1, y0, y1 = self._fraction_bounds(_SMART_NO_FACE_FALLBACK, w, h)
            else:
                x0, x1, y0, y1 = self._smart_format_bounds(face, w)

        crop = card[y0:y1, x0:x1]
        if crop.size == 0:
            logger.warning(f"Bangla name extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Bangla name extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()

    @staticmethod
    def _fraction_bounds(region: dict, w: int, h: int) -> tuple[int, int, int, int]:
        return int(w * region["x0"]), int(w * region["x1"]), int(h * region["y0"]), int(h * region["y1"])

    @staticmethod
    def _smart_format_bounds(face: tuple[int, int, int, int], w: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        x0 = int(fx + fw + fw * _SMART_RIGHT_GAP)
        y0 = max(0, int(fy - fh * _SMART_ABOVE_FACE))
        y1 = int(y0 + fh * _SMART_ROW_HEIGHT)
        return x0, w, y0, y1
