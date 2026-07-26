import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.photo_locator import detect_largest_face
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# The Bengali "নাম:" line sits in the text column to the right of the photo,
# ending roughly where the face box starts (the face box brackets the
# "Name:"-through-"মাতা:" rows, not নাম itself). Rather than branching on
# card format — which turned out to correlate with label wording ("NID No"
# vs "ID NO"), not with where নাম actually sits — just crop a generously
# wide band above the face for every card. Oversized on purpose: better to
# include a bit of the header/English-name line than to miss নাম.
_RIGHT_GAP = 0.10        # of face width, gap between photo and text column
_ABOVE_FACE = 0.70       # of face height, how far above the face top the crop starts
_BELOW_FACE_TOP = 0.10   # of face height, how far past the face top the crop extends

# Fallback region if no face is detected at all.
_NO_FACE_FALLBACK = dict(x0=0.28, x1=0.98, y0=0.05, y1=0.45)


class BanglaNameExtractor:
    def extract(self, image_path: str) -> bytes | None:
        """Return the cropped Bengali name-line region as PNG bytes, or None
        if the image couldn't be read or the crop came out empty."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Bangla name extraction: could not read {image_path}")
            return None

        card = locate_card(img)
        h, w = card.shape[:2]

        face = detect_largest_face(card)
        if face is None:
            logger.info(f"Bangla name extraction: no face found, using fallback region for {image_path}")
            x0, x1, y0, y1 = self._fraction_bounds(_NO_FACE_FALLBACK, w, h)
        else:
            x0, x1, y0, y1 = self._face_relative_bounds(face, w)

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
    def _face_relative_bounds(face: tuple[int, int, int, int], w: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        x0 = int(fx + fw + fw * _RIGHT_GAP)
        y0 = max(0, int(fy - fh * _ABOVE_FACE))
        y1 = int(fy + fh * _BELOW_FACE_TOP)
        return x0, w, y0, y1
