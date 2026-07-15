import cv2
import numpy as np

# ID-1 card physical ratio (85.60mm x 53.98mm) — a correctly-detected NID
# card boundary should be close to this regardless of how the card was
# framed/rotated in the uploaded photo.
_ID1_RATIO = 85.60 / 53.98
_MIN_AREA_FRACTION = 0.15
_MIN_QUAD_SCORE = 0.5


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")


def find_card_quad(img: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Locate the NID card's rectangular boundary in `img`.

    Returns the 4 ordered corners (tl, tr, br, bl) and a confidence score, or
    (None, -1) if no candidate contour resembles an ID-1 card closely enough
    — which also covers uploads that are already a tight, near-fullframe crop
    of just the card, since there's no background edge to find there.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    img_area = h * w
    best_quad, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_AREA_FRACTION * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.int32)
        quad = _order_corners(approx)

        side_w = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2
        side_h = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2
        if side_h == 0:
            continue
        ratio = side_w / side_h
        ratio = ratio if ratio >= 1 else 1 / ratio
        ratio_score = 1 - min(abs(ratio - _ID1_RATIO) / _ID1_RATIO, 1)
        area_score = area / img_area
        score = ratio_score * 0.7 + area_score * 0.3

        if score > best_score:
            best_quad, best_score = quad, score

    if best_score < _MIN_QUAD_SCORE:
        return None, -1.0
    return best_quad, best_score


def warp_card(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct `img` to an upright, cropped card using `quad`."""
    tl, tr, br, bl = quad
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, matrix, (max_w, max_h))


def locate_card(img: np.ndarray) -> np.ndarray:
    """Return an upright, cropped canonical card image.

    Falls back to the original image unchanged when no card boundary can be
    confidently detected (see find_card_quad), since that case already means
    the upload is essentially just the card with little to no background.
    """
    quad, _ = find_card_quad(img)
    if quad is None:
        return img
    return warp_card(img, quad)
