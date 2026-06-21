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

    def preprocess(self, input_path: str, output_dir: str) -> dict[str, str]:
        """
        Returns a dict of variant_name → file_path for each preprocessed version.
        Two variants are produced:
          - v1_clahe: contrast-enhanced grayscale (good for faded/low-contrast)
          - v3_sharp: CLAHE + sharpened (good for blurry images)
        """
        img = cv2.imread(input_path)
        if img is None:
            raise ImageReadError(f"Cannot read image: {input_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None,
                          fx=self._cfg.upscale_factor,
                          fy=self._cfg.upscale_factor,
                          interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, h=self._cfg.denoise_h)

        clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip_limit,
            tileGridSize=self._cfg.clahe_tile_grid
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
            logger.info(f"Saved preprocessed variant '{name}' → {path}")

        return paths
