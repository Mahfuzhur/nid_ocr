_BANGLA_TO_ASCII = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


def convert_bangla_digits(text: str) -> str:
    return text.translate(_BANGLA_TO_ASCII)
