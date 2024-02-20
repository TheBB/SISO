from typing import Union

import numpy as np
import numpy.typing as npt

class Rotation:
    @classmethod
    def from_euler(cls, seq: str, angles: Union[float, npt.ArrayLike], degrees: bool = ...) -> Rotation: ...
    def inv(self) -> Rotation: ...
    def apply(self, vectors: npt.ArrayLike, inverse: bool = ...) -> np.ndarray: ...
