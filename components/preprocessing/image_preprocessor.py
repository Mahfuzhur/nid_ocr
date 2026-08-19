import os
import cv2
import numpy as np
from nid_ocr.core.config import Settings
from nid_ocr.core.exceptions import ImageReadError
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Produces multiple preprocessed image variants to maximize OCR accuracy
    across different NID card types, image qualities, and lighting conditions."""

    def __init__(self, settings: Settings):
        self._cfg = settings

    def preprocess(self, input_path: str, output_dir: str, mode: str = 'front') -> dict[str, str]:
        """
        Returns variant_name → file_path for each preprocessed version.

        Variants produced:
          v1_clahe — contrast-enhanced grayscale (faded/low-contrast images)
          v3_sharp — CLAHE + sharpened (blurry images)

        `mode` is accepted but currently both card sides use the same variants.
        NID cards have patterned security backgrounds that binarization-based
        variants (Otsu, adaptive) amplify as noise, so they are not included.
        """
        img = cv2.imread(input_path)
        if img is None:
            raise ImageReadError(f"Cannot read image: {input_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        long_side = max(h, w)
        target = int(long_side * self._cfg.upscale_factor)
        # Cap at max_long_side to avoid runaway memory/time on high-res source images
        capped = min(target, self._cfg.max_long_side)
        scale = capped / long_side
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, h=self._cfg.denoise_h)

        clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip_limit,
            tileGridSize=self._cfg.clahe_tile_grid,
        )
        v1 = clahe.apply(gray)

        sharp_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        v3 = cv2.filter2D(v1, -1, sharp_kernel)

        paths: dict[str, str] = {}
        for name, img_data in [('v1_clahe', v1), ('v3_sharp', v3)]:
            path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(path, img_data)
            paths[name] = path
            logger.info(f"Prepared preprocessed variant '{name}'")

        return paths
