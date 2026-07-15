import cv2
import numpy as np

# Relative to full card area — filters out tiny noise specks (dust,
# background pattern fragments) that survive thresholding.
_MIN_COMPONENT_AREA_FRACTION = 0.0005
# A candidate cluster further from the anchor than this (relative to the
# anchor region's own diagonal) is treated as unrelated content, not a
# mis-positioned signature.
_MAX_ANCHOR_DISTANCE_FACTOR = 1.5
# Guards against picking a cluster that has merged with the photo or another
# text block (e.g. if dilation bridged a whitespace gap that shouldn't have
# been bridged) — such a cluster would be implausibly large next to the
# calibrated anchor region.
_MAX_BBOX_AREA_FACTOR = 6.0
_PADDING = 10


def find_signature_bbox(
    card: np.ndarray, anchor: tuple[float, float, float, float]
) -> tuple[int, int, int, int] | None:
    """Find a tight pixel bounding box (x0, y0, x1, y1) around the ink cluster
    nearest the calibrated anchor region on `card`, growing to include the
    full signature but stopping at whitespace or another text block — rather
    than using a fixed-percentage crop, which can clip part of the signature
    for cards where it sits slightly outside the calibrated region.

    `anchor` is (x0, y0, x1, y1) as a fraction of the card's width/height —
    used only to pick the right connected component (nearest centroid, with
    sanity bounds on distance/size), not as the crop bounds themselves.
    Returns None if no plausible cluster is found near the anchor.
    """
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    # Inverted so ink = white(255), background = black(0). Otsu's threshold
    # cleanly separates printed/handwritten ink from the card's light green
    # background and decorative security pattern.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bridge small gaps between cursive signature strokes into one connected
    # component, without merging separate text lines/blocks that have real
    # whitespace between them (wider than tall: signatures are horizontal).
    kernel = np.ones((9, 25), np.uint8)
    dilated = cv2.dilate(th, kernel, iterations=1)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    ax0, ay0, ax1, ay1 = anchor
    anchor_cx, anchor_cy = (ax0 + ax1) / 2 * w, (ay0 + ay1) / 2 * h
    anchor_area = (ax1 - ax0) * w * (ay1 - ay0) * h
    anchor_diag = (((ax1 - ax0) * w) ** 2 + ((ay1 - ay0) * h) ** 2) ** 0.5

    best_label, best_dist = None, None
    for i in range(1, n):  # label 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < _MIN_COMPONENT_AREA_FRACTION * h * w:
            continue
        if area > anchor_area * _MAX_BBOX_AREA_FACTOR:
            continue
        cx, cy = centroids[i]
        dist = ((cx - anchor_cx) ** 2 + (cy - anchor_cy) ** 2) ** 0.5
        if dist > anchor_diag * _MAX_ANCHOR_DISTANCE_FACTOR:
            continue
        if best_dist is None or dist < best_dist:
            best_dist, best_label = dist, i

    if best_label is None:
        return None

    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    x0, y0 = max(0, x - _PADDING), max(0, y - _PADDING)
    x1, y1 = min(w, x + bw + _PADDING), min(h, y + bh + _PADDING)
    return x0, y0, x1, y1
