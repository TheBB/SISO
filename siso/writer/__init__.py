import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

from .api import Writer


class OutputFormat(Enum):
    Vtf = "vtf"
    Vtk = "vtk"
    Vtu = "vtu"
    Vts = "vts"
    Pvd = "pvd"
    Simra = "simra"

    def default_suffix(self):
        if self == OutputFormat.Simra:
            return '.dat'
        return f".{self.value}"


MakeWriter = Callable[[Path], Writer]

WRITERS: Dict[OutputFormat, MakeWriter] = {}


def register_writer(fmt: OutputFormat) -> Callable[[MakeWriter], MakeWriter]:
    def inner(constructor: MakeWriter) -> MakeWriter:
        WRITERS[fmt] = constructor
        return constructor

    return inner


def find_writer(fmt: OutputFormat, path: Path) -> Optional[Writer]:
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
