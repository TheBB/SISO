from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeStep:
    index: int
    time: Optional[float] = None
