import cv2

_face_cascade = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def detect_largest_face(card) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest detected face on `card`, or None.

    Largest by area — filters false positives (e.g. the government emblem)
    that Haar cascades occasionally trigger on elsewhere on the card.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))
