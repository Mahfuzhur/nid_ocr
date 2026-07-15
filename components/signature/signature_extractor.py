import cv2

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.photo_locator import detect_largest_face
from nid_ocr.components.signature.ink_bounds import find_signature_row_band, find_ink_right_edge
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# SMART (chip) cards: proven stable — the photo's right/bottom margins as a
# fraction of the detected face box, and the crop runs all the way to the
# card's bottom edge (there's reliably clear whitespace there).
_SMART_MARGINS = dict(right=0.12, below=0.17)

# OLD (laminated) cards pack "Date of Birth"/"ID NO" text much closer beneath
# the signature — different OLD templates by different margins, so no single
# fixed ratio works for all of them (confirmed against two real cards with
# very different needed margins). find_signature_row_band/find_ink_right_edge
# locate the actual ink content instead. These fixed ratios are only the
# fallback if that content-based search doesn't find a plausible signature.
_OLD_FALLBACK = dict(right=0.45, below=0.30, height=0.06)

# Fallback region if no face is detected at all, for either format.
_NO_FACE_FALLBACK = {
    NIDFormat.SMART: dict(x0=0.03, y0=0.80, y1=0.99, x1=0.30),
    NIDFormat.OLD:   dict(x0=0.02, y0=0.68, y1=0.98, x1=0.25),
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

        face = detect_largest_face(card)
        if face is None:
            logger.info(f"Signature extraction: no face found, using fallback region for {image_path}")
            region = _NO_FACE_FALLBACK.get(fmt, _NO_FACE_FALLBACK[NIDFormat.SMART])
            x0, x1 = int(w * region["x0"]), int(w * region["x1"])
            y0, y1 = int(h * region["y0"]), int(h * region["y1"])
        elif fmt == NIDFormat.OLD:
            x0, x1, y0, y1 = self._old_format_bounds(card, face, w, h)
        else:
            x0, x1, y0, y1 = self._smart_format_bounds(face, h)

        crop = card[y0:y1, x0:x1]
        if crop.size == 0:
            logger.warning(f"Signature extraction: empty crop for {image_path}")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning(f"Signature extraction: PNG encode failed for {image_path}")
            return None
        return buf.tobytes()

    @staticmethod
    def _smart_format_bounds(face: tuple[int, int, int, int], h: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        x1 = int(fx + fw + fw * _SMART_MARGINS["right"])
        y0 = int(fy + fh + fh * _SMART_MARGINS["below"])
        return 0, x1, y0, h  # run to the card's bottom edge

    def _old_format_bounds(self, card, face, w: int, h: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        face_bottom = fy + fh
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

        band = find_signature_row_band(gray, x1=fx + fw, y_start=face_bottom, y_end=h)
        if band is not None:
            sig_start, sig_end = band
            y0 = max(0, face_bottom + sig_start - 4)
            y1 = min(h, face_bottom + sig_end + 3)

            x1_search = min(w, int(w * 0.55))
            x1 = find_ink_right_edge(gray[y0:y1, 0:x1_search]) + 6
            x1 = min(x1_search, x1)
            return 0, x1, y0, y1

        logger.info("Signature extraction: row-band search failed, using OLD-format fallback ratios")
        x1 = int(fx + fw + fw * _OLD_FALLBACK["right"])
        y0 = int(face_bottom + fh * _OLD_FALLBACK["below"])
        y1 = min(h, int(y0 + h * _OLD_FALLBACK["height"]))
        return 0, x1, y0, y1
