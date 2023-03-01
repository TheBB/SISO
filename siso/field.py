from __future__ import annotations

from dataclasses import dataclass

from . import api


@dataclass
class Field(api.Field):
    name: str
    type: api.FieldType
    cellwise: bool = False
    splittable: bool = True
