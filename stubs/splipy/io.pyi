from types import TracebackType
from typing import Optional, TextIO

from splipy import SplineObject
from typing_extensions import Self

class G2:
    fstream: TextIO
    onlywrite: bool
    def __init__(self, filename: str) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    def read(self) -> list[SplineObject]: ...
