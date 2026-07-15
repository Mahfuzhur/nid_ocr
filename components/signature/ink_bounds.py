import cv2
import numpy as np

# Used by the row scan to tell "inside the photo" (high pixel variance: face,
# hair, clothing texture) apart from "background" (low, uniform variance) —
# expressed as a fraction of the variance measured right at the photo's
# start, so it adapts to each card's own contrast/lighting rather than an
# absolute number.
_PHOTO_END_FRACTION = 0.4
# How far above the local background baseline a row must rise to count as
# the signature starting, and how close back to it to count as ending.
_SIGNATURE_RISE_FACTOR = 3.0
_SIGNATURE_RISE_MARGIN = 8.0
_SIGNATURE_FALL_FACTOR = 1.5
_SIGNATURE_FALL_MARGIN = 3.0

_INK_THRESHOLD = 2
_COL_MIN_START_RUN = 3
_COL_GAP_MIN = 6
_COL_GAP_FRACTION = 0.04


def find_signature_row_band(
    gray: np.ndarray, x1: int, y_start: int, y_end: int
) -> tuple[int, int] | None:
    """Within gray[y_start:y_end, 0:x1], find the signature's (y0, y1) as
    offsets from y_start, using each row's pixel standard deviation.

    Real ID photos are followed by a fairly uniform background before the
    next printed field, so scanning down from the photo shows a distinctive
    pattern: high variance (photo) -> a low/uniform baseline (background) ->
    a spike back up (the signature's ink) -> back to baseline (background
    again, before whatever text comes next). Thresholds are relative to each
    card's own measured variance rather than fixed values, since different
    print templates have very different background/ink contrast.
    Returns None if that pattern isn't found (caller should fall back).
    """
    rows = gray[y_start:y_end, 0:x1].astype(np.float64)
    if rows.shape[0] < 10:
        return None
    row_std = rows.std(axis=1)

    photo_std = row_std[:5].mean()
    if photo_std <= 0:
        return None

    photo_end = None
    for i in range(len(row_std) - 1):
        if row_std[i] < _PHOTO_END_FRACTION * photo_std and row_std[i + 1] < _PHOTO_END_FRACTION * photo_std:
            photo_end = i
            break
    if photo_end is None:
        return None

    baseline = max(row_std[photo_end:photo_end + 5].mean(), 0.5)
    rise_thresh = max(_SIGNATURE_RISE_FACTOR * baseline, baseline + _SIGNATURE_RISE_MARGIN)

    sig_start = None
    for i in range(photo_end, len(row_std)):
        if row_std[i] > rise_thresh:
            sig_start = i
            break
    if sig_start is None:
        return None

    fall_thresh = max(_SIGNATURE_FALL_FACTOR * baseline, baseline + _SIGNATURE_FALL_MARGIN)
    sig_end = len(row_std)
    for i in range(sig_start, len(row_std) - 1):
        if row_std[i] < fall_thresh and row_std[i + 1] < fall_thresh:
            sig_end = i
            break

    return sig_start, sig_end


def find_ink_right_edge(gray_band: np.ndarray) -> int:
    """Return the column where ink in `gray_band` ends: the start of the
    first real whitespace gap after it begins. Lets the crop grow to
    whatever width the actual content needs, stopping at the first genuine
    gap — e.g. before the next printed field — rather than a fixed width
    that either clips wide content or, if widened generically, risks
    reaching into that next field.
    """
    _, th = cv2.threshold(gray_band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_ink = th.sum(axis=0) / 255
    n = len(col_ink)

    run, started_at = 0, None
    for i, v in enumerate(col_ink):
        if v > _INK_THRESHOLD:
            run += 1
            if run >= _COL_MIN_START_RUN and started_at is None:
                started_at = i - run + 1
        else:
            run = 0
    if started_at is None:
        return n

    gap_needed = max(_COL_GAP_MIN, int(_COL_GAP_FRACTION * n))
    gap_run = 0
    for i in range(started_at, n):
        if col_ink[i] > _INK_THRESHOLD:
            gap_run = 0
        else:
            gap_run += 1
            if gap_run >= gap_needed:
                return i - gap_run + 1
    return n
