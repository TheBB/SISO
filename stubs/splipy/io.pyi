from types import TracebackType
from typing import TextIO

from splipy import SplineObject
from typing_extensions import Self

class G2:
    fstream: TextIO
    onlywrite: bool
    def __init__(self, filename: str) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def read(self) -> list[SplineObject]: ...
