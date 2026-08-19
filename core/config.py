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
    max_long_side: int = field(default_factory=lambda: int(os.environ.get('MAX_LONG_SIDE', '2800')))

    # OCR — engine used for all requests; override via DEFAULT_OCR_ENGINE env var.
    default_ocr_engine: str = field(default_factory=lambda: os.environ.get('DEFAULT_OCR_ENGINE', 'surya'))
    # Surya dtype: 'float32' (full precision) or 'float16' (Surya GPU default).
    # float32 → ~2x GPU memory, ~30% slower, better accuracy.
    # CPU deployments run float32 regardless of this value.
    surya_dtype: str = field(default_factory=lambda: os.environ.get('SURYA_DTYPE', 'float16'))
    surya_recognition_batch_size: int = field(
        default_factory=lambda: int(os.environ.get('SURYA_RECOGNITION_BATCH_SIZE', '64'))
    )
    surya_detector_batch_size: int = field(
        default_factory=lambda: int(os.environ.get('SURYA_DETECTOR_BATCH_SIZE', '16'))
    )
    ocr_max_concurrency: int = field(default_factory=lambda: int(os.environ.get('OCR_MAX_CONCURRENCY', '2')))
    ocr_queue_wait_seconds: float = field(
        default_factory=lambda: float(os.environ.get('OCR_QUEUE_WAIT_SECONDS', '5'))
    )
    ocr_warmup_enabled: bool = field(
        default_factory=lambda: os.environ.get('OCR_WARMUP_ENABLED', 'true').lower() in {'1', 'true', 'yes'}
    )
    easyocr_languages: list = field(default_factory=lambda: ['bn', 'en'])
    easyocr_gpu: bool = False
    easyocr_min_confidence: float = 0.0
    tesseract_lang: str = 'ben+eng'
    tesseract_config: str = '--psm 6'
    paddle_lang: str = 'en'

    # Image upload
    allowed_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # Permanent storage — every uploaded image is copied here (organized by date),
    # in addition to the temp copy used for processing. Override via UPLOAD_STORAGE_DIR.
    upload_storage_dir: str = field(default_factory=lambda: os.environ.get('UPLOAD_STORAGE_DIR', 'storage/uploads'))

    # MySQL — stores one row per upload (front or back) with the extracted fields,
    # so the /uploads page can list and filter past results.
    db_host: str = field(default_factory=lambda: os.environ.get('DB_HOST', 'localhost'))
    db_port: int = field(default_factory=lambda: int(os.environ.get('DB_PORT', '3306')))
    db_user: str = field(default_factory=lambda: os.environ.get('DB_USER', 'root'))
    db_password: str = field(default_factory=lambda: os.environ.get('DB_PASSWORD', ''))
    db_name: str = field(default_factory=lambda: os.environ.get('DB_NAME', 'nid_ocr'))

    # IndicNLP
    indic_resources_path: str = os.environ.get('INDIC_RESOURCES_PATH', '')


settings = Settings()
