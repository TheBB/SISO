from typing import Any, Tuple, Tuple, Hashable

from nptyping import NDArray


Array2D = NDArray[Any, Any]
BoundingBox = Tuple[Tuple[float, float], ...]
PatchID = Tuple[Hashable, ...]
