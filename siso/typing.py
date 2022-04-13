from typing import Any, Tuple, Tuple, Hashable, Dict, Iterable

from nptyping import NDArray, Shape


Array1D = NDArray[Shape['*'], Any]
Array2D = NDArray[Shape['*,*'], Any]
Array = NDArray[Shape['*,...'], Any]

BoundingBox = Tuple[Tuple[float, float], ...]
Knots = Tuple[Tuple[float, ...], ...]
PatchKey = Tuple[Hashable, ...]
Shape = Tuple[int, ...]
StepData = Dict[str, int]
