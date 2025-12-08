from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from . import api

if TYPE_CHECKING:
    from siso.types import Float


@define(frozen=True)
class Basis(api.Basis):
    name: str


@define(frozen=True)
class Step(api.Step):
    index: int
    value: Float | None = None


@define(frozen=True)
class Field(api.Field):
    name: str
    type: api.FieldType
    cellwise: bool = field(default=False, kw_only=True)
    splittable: bool = field(default=True, kw_only=True)
