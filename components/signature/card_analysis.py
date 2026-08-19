from dataclasses import dataclass

import cv2
import numpy as np

from nid_ocr.components.signature.card_locator import locate_card
from nid_ocr.components.signature.photo_locator import detect_largest_face
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FrontCardAnalysis:
    card: np.ndarray
    face: tuple[int, int, int, int] | None


def analyze_front_card(image_path: str) -> FrontCardAnalysis | None:
    """Read, locate and inspect the card once for all front-side crop outputs."""
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Front card analysis: could not read image")
        return None
    card = locate_card(image)
    return FrontCardAnalysis(card=card, face=detect_largest_face(card))
