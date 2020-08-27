from typing import Any, Tuple, Tuple, Hashable, Dict

from nptyping import NDArray


Array1D = NDArray[Any]
Array2D = NDArray[Any, Any]

BoundingBox = Tuple[Tuple[float, float], ...]
Knots = Tuple[Tuple[float, ...], ...]
PatchKey = Tuple[Hashable, ...]
Shape = Tuple[int, ...]
StepData = Dict[str, int]
