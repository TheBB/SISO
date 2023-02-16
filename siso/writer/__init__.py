from enum import Enum
import logging
from pathlib import Path

from .api import Writer

from typing import Callable, Dict, Optional, TYPE_CHECKING


class OutputFormat(Enum):
    Vtf = 'vtf'
    Vtk = 'vtk'
    Vtu = 'vtu'
    Vts = 'vts'
    Pvd = 'pvd'

    def default_suffix(self):
        return f'.{self.value}'


MakeWriter = Callable[[Path], Writer]

WRITERS: Dict[OutputFormat, MakeWriter] = {}

def register_writer(fmt: OutputFormat):
    def inner(constructor: MakeWriter):
        WRITERS[fmt] = constructor
    return inner

def find_writer(fmt: OutputFormat, path: Path) -> Optional[Writer]:
    constructor = WRITERS[fmt]
    try:
        return constructor(path)
    except ImportError:
        logging.critical(f'Unable to write {fmt} format - some dependencies may not be installed')
        return None


@register_writer(OutputFormat.Vtk)
def vtk(path: Path) -> Writer:
    from .vtk import VtkWriter
    return VtkWriter(path)
