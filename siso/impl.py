from typing import Optional

from attrs import define

from . import api


@define
class Basis(api.Basis):
    name: str


@define(eq=False)
class Step(api.Step):
    index: int
    value: Optional[float] = None


@define(eq=False)
class Field(api.Field):
    name: str
    type: api.FieldType
    cellwise: bool = False
    splittable: bool = True
