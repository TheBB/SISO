from typing import Literal

from numpy import ndarray

def reshape(
    cps: ndarray,
    newshape: tuple[int, ...],
    order: Literal["F", "C"] = "C",
    ncomps: int | None = None,
) -> ndarray: ...
