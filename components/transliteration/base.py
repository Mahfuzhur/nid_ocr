from abc import ABC, abstractmethod


class Transliterator(ABC):
    @abstractmethod
    def transliterate(self, text: str) -> str:
        ...
