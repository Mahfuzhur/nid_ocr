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

    # OCR
    easyocr_languages: list = field(default_factory=lambda: ['bn', 'en'])
    easyocr_gpu: bool = False
    tesseract_lang: str = 'ben+eng'
    tesseract_config: str = '--psm 6'

    # Image upload
    allowed_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # IndicNLP
    indic_resources_path: str = os.environ.get('INDIC_RESOURCES_PATH', '')


settings = Settings()
