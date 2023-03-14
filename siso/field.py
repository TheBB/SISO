from __future__ import annotations

from attrs import define

from . import api


@define(eq=False)
class Field(api.Field):
    name: str
    type: api.FieldType
    cellwise: bool = False
    splittable: bool = True
