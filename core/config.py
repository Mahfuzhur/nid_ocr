import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Preprocessing
    upscale_factor: float = 4.0
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: tuple = (8, 8)
    denoise_h: int = 15
    adaptive_block_size: int = 31
    adaptive_c: int = 10

    # If source image * upscale_factor exceeds this long-side pixel count, cap it.
    # Prevents EasyOCR from running on 100MP+ images when input is already high-res.
    max_long_side: int = 2000

    # OCR
    easyocr_languages: list = field(default_factory=lambda: ['bn', 'en'])
    easyocr_gpu: bool = False
    easyocr_min_confidence: float = 0.0
    tesseract_lang: str = 'ben+eng'
    tesseract_config: str = '--psm 6'
    paddle_lang: str = 'en'

    # Image upload
    allowed_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # IndicNLP
    indic_resources_path: str = os.environ.get('INDIC_RESOURCES_PATH', '')


settings = Settings()
