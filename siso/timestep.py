from typing import Optional

from attrs import define


@define(eq=False)
class Step:
    index: int
    value: Optional[float] = None
