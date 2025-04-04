from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, TypeVar, overload

from numpy import ndarray

T = TypeVar("T")

class Group:
    def __contains__(self, path: str) -> bool: ...
    @overload
    def __getitem__(self, i: str) -> Group: ...
    @overload
    def __getitem__(self, i: slice) -> ndarray: ...
    @overload
    def __getitem__(self, i: int) -> float: ...
    def __iter__(self) -> Iterator[str]: ...
    def values(self) -> Iterator[Group]: ...
    def items(self) -> Iterator[tuple[str, Group]]: ...
    def get(self, name: str, default: T) -> T | Group: ...
    def __len__(self) -> int: ...

class File(Group):
    def __init__(self, filename: Path, mode: Literal["r", "w"] = ...) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
