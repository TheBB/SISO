from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from .api import Writer


class OutputFormat(Enum):
    Vtf = "vtf"
    Vtk = "vtk"
    Vtu = "vtu"
    Vts = "vts"
    Pvd = "pvd"
    Simra = "simra"

    def default_suffix(self) -> str:
        if self == OutputFormat.Simra:
            return ".dat"
        return f".{self.value}"


MakeWriter = Callable[[Path], Writer]

WRITERS: dict[OutputFormat, MakeWriter] = {}


def register_writer(fmt: OutputFormat) -> Callable[[MakeWriter], MakeWriter]:
    def inner(constructor: MakeWriter) -> MakeWriter:
        WRITERS[fmt] = constructor
        return constructor

    return inner


def find_writer(fmt: OutputFormat, path: Path) -> Writer | None:
    constructor = WRITERS[fmt]
    try:
        return constructor(path)
    except ImportError:
        logging.critical(f"Unable to write {fmt} format - some dependencies may not be installed")
        return None


@register_writer(OutputFormat.Vtk)
def vtk(path: Path) -> Writer:
    from .vtk import VtkWriter

    return VtkWriter(path)


@register_writer(OutputFormat.Vtu)
def vtu(path: Path) -> Writer:
    from .vtk import VtuWriter

    return VtuWriter(path)


@register_writer(OutputFormat.Vts)
def vts(path: Path) -> Writer:
    from .vtk import VtsWriter

    return VtsWriter(path)


@register_writer(OutputFormat.Pvd)
def pvd(path: Path) -> Writer:
    from .vtk import PvdWriter

    return PvdWriter(path)


@register_writer(OutputFormat.Simra)
def simra(path: Path) -> Writer:
    from .simra import SimraWriter

    return SimraWriter(path)


@register_writer(OutputFormat.Vtf)
def vtf(path: Path) -> Writer:
    from .vtf import VtfWriter

    return VtfWriter(path)
