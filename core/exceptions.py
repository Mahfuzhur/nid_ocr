class NIDOCRError(Exception):
    pass


class ImageReadError(NIDOCRError):
    pass


class OCRError(NIDOCRError):
    pass


class ExtractionError(NIDOCRError):
    pass
