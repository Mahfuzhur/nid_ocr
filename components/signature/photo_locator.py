import cv2

from nid_ocr.domain.enums import NIDFormat

# Portrait photo occupies the card's left edge on every Bangladesh NID front
# template, so its left margin is always ~0. Its right and bottom margins are
# estimated from the detected face, scaled by face size — the face itself is
# a far more reliable thing to detect than the (lower-contrast, sometimes
# background-blended) edges of the printed photo box. Ratios are calibrated
# per format since SMART (chip) and OLD (laminated) cards use different
# physical layouts/proportions.
_FACE_MARGINS = {
    NIDFormat.SMART: dict(right=0.14, below=0.17),
    NIDFormat.OLD:   dict(right=0.50, below=0.30),
}

_face_cascade = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def locate_photo_margins(card, fmt: NIDFormat) -> tuple[int, int, int] | None:
    """Return (left, right, bottom) pixel margins of the ID photo on `card`,
    or None if no face is found. left is always 0 (photo sits at the card's
    left edge); right/bottom are derived from the detected face box, scaled
    by format-specific margins calibrated against the real card template.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    # Largest detected face by area — filters false positives (e.g. the
    # government emblem) that Haar cascades occasionally trigger on.
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    margins = _FACE_MARGINS.get(fmt, _FACE_MARGINS[NIDFormat.SMART])
    right = int(fx + fw + fw * margins["right"])
    bottom = int(fy + fh + fh * margins["below"])
    return 0, right, bottom
