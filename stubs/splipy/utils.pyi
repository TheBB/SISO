from typing import Literal, Optional

from numpy import ndarray

def reshape(
    cps: ndarray,
    newshape: tuple[int, ...],
    order: Literal["F", "C"] = "C",
    ncomps: Optional[int] = None,
) -> ndarray: ...
