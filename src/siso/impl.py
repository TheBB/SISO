from __future__ import annotations

from attrs import define, field

from . import api


@define(frozen=True)
class Basis(api.Basis):
    name: str


@define(frozen=True)
class Step(api.Step):
    index: int
    value: float | None = None


@define(frozen=True)
class Field(api.Field):
    name: str
    type: api.FieldType
    cellwise: bool = field(default=False, kw_only=True)
    splittable: bool = field(default=True, kw_only=True)
