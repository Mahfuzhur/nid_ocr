import cv2

from nid_ocr.components.signature.card_analysis import FrontCardAnalysis, analyze_front_card
from nid_ocr.domain.enums import NIDFormat
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

# SMART (chip) cards: proven stable — the photo's right/bottom margins as a
# fraction of the detected face box, and the crop runs all the way to the
# card's bottom edge (there's reliably clear whitespace there).
_SMART_MARGINS = dict(right=0.12, below=0.17)

# OLD (laminated) cards: different OLD templates need different margins, and
# content-based ink scanning (row-variance band search, column-gap search)
# proved unreliable across real uploads — false-positive bands, seams left
# by card_locator's perspective warp misread as ink, search-width ceilings
# silently accepted as real edges. Replaced with a fixed box sized as a
# fraction of the card's own detected width/height, measured against
# hand-boxed reference signature crops on real cards and consistent
# regardless of the photo's resolution or framing. Only the vertical anchor
# (below the detected face) still depends on the face box.
_OLD_FALLBACK = dict(below=0.30)
_CARD_WIDTH_FRACTION = 0.28  # of card width, anchored at x=0
_CARD_HEIGHT_FRACTION = 0.30  # of card height

# Fallback region if no face is detected at all, for either format.
_NO_FACE_FALLBACK = {
    NIDFormat.SMART: dict(x0=0.03, y0=0.80, y1=0.99, x1=0.30),
    NIDFormat.OLD:   dict(x0=0.02, y0=0.68, y1=0.98, x1=0.25),
}


class SignatureExtractor:
    def extract(self, image_path: str, fmt: NIDFormat) -> bytes | None:
        """Return the cropped signature region as PNG bytes, or None if the
        image couldn't be read or the crop came out empty."""
        analysis = analyze_front_card(image_path)
        return self.extract_from_analysis(analysis, fmt)

    def extract_from_analysis(self, analysis: FrontCardAnalysis | None, fmt: NIDFormat) -> bytes | None:
        if analysis is None:
            return None
        card = analysis.card
        h, w = card.shape[:2]

        face = analysis.face
        if face is None:
            logger.info("Signature extraction: no face found, using fallback region")
            region = _NO_FACE_FALLBACK.get(fmt, _NO_FACE_FALLBACK[NIDFormat.SMART])
            x0, x1 = int(w * region["x0"]), int(w * region["x1"])
            y0, y1 = int(h * region["y0"]), int(h * region["y1"])
        elif fmt == NIDFormat.OLD:
            x0, x1, y0, y1 = self._old_format_bounds(face, w, h)
        else:
            x0, x1, y0, y1 = self._smart_format_bounds(face, h)

        crop = card[y0:y1, x0:x1]
        if crop.size == 0:
            logger.warning("Signature extraction: empty crop")
            return None

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            logger.warning("Signature extraction: PNG encode failed")
            return None
        return buf.tobytes()

    @staticmethod
    def _smart_format_bounds(face: tuple[int, int, int, int], h: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        x1 = int(fx + fw + fw * _SMART_MARGINS["right"])
        y0 = int(fy + fh + fh * _SMART_MARGINS["below"])
        return 0, x1, y0, h  # run to the card's bottom edge

    @staticmethod
    def _old_format_bounds(face: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = face
        face_bottom = fy + fh
        x1 = int(w * _CARD_WIDTH_FRACTION)
        y0 = int(face_bottom + fh * _OLD_FALLBACK["below"])
        y1 = min(h, int(y0 + h * _CARD_HEIGHT_FRACTION))
        return 0, x1, y0, y1
