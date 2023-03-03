from typing import Optional

from attrs import define


@define
class TimeStep:
    index: int
    time: Optional[float] = None
