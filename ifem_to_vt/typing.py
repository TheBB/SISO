from typing import Any, Tuple, Tuple, Hashable, Dict

from nptyping import NDArray


Array2D = NDArray[Any, Any]
BoundingBox = Tuple[Tuple[float, float], ...]
PatchID = Tuple[Hashable, ...]
Shape = Tuple[int, ...]
StepData = Dict[str, int]
