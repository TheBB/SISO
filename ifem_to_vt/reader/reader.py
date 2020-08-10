from abc import ABC
from pathlib import Path

import treelog as log

from ..util import subclasses


class Reader(ABC):

    reader_name: str

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        """Return true if the class can handle files of the given type."""
        return False

    @staticmethod
    def find_applicable(filename: Path) -> type:
        """Return a reader subclass that can handle files of the given type."""
        for cls in subclasses(Reader, invert=True):
            if cls.applicable(filename):
                log.info(f"Found applicable reader: {cls.reader_name}")
                return cls
            else:
                log.debug(f"Rejecting reader: {cls.reader_name}")
        raise TypeError(f"Unable to find any applicable readers for {filename}")
