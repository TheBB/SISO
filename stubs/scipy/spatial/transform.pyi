import numpy as np
import numpy.typing as npt

class Rotation:
    @classmethod
    def from_euler(cls, seq: str, angles: float | npt.ArrayLike, degrees: bool = ...) -> Rotation: ...
    def inv(self) -> Rotation: ...
    def apply[S: tuple[int, ...]](
        self,
        vectors: np.ndarray[S, np.dtype[np.float32]] | np.ndarray[S, np.dtype[np.float64]],
        inverse: bool = ...,
    ) -> np.ndarray[S, np.dtype[np.float32]] | np.ndarray[S, np.dtype[np.float64]]: ...
