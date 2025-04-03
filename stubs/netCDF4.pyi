from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from numpy.ma import masked_array

class Dimension: ...

class Variable:
    dimensions: Sequence[str]
    def __getitem__(self, index: Any) -> masked_array: ...

class Dataset:
    dimensions: dict[str, Sequence[Dimension]]

    def __init__(self, filename: Path, mode: Literal["r", "w"]) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def __getitem__(self, name: str) -> Variable: ...
    def __getattr__(self, name: str) -> Any: ...
