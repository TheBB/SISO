from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, Self

from numpy import dtype, ndarray

class FortranEOFError(TypeError, OSError): ...
class FortranFormattingError(TypeError, OSError): ...

class FortranFile:
    _fp: BinaryIO
    def __init__(self, filename: Path, mode: Literal["r", "w"], header_dtype: dtype) -> None: ...
    def _read_size(self, eof_ok: bool = ...) -> int: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def write_record(self, *items: ndarray) -> None: ...
    def read_ints(self, dtype: dtype) -> Sequence[int]: ...
    def read_reals(self, dtype: dtype) -> ndarray: ...
