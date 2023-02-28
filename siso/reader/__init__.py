from dataclasses import dataclass
import logging
from pathlib import Path

from typing import Callable, List, Tuple, Optional

from ..api import Endianness, Source


@dataclass
class FindReaderSettings:
    endianness: Endianness


ReaderCheck = Callable[[Path, FindReaderSettings], Optional[Source]]

READERS: List[Tuple[str, ReaderCheck]] = []

def register_reader(name: str) -> Callable[[ReaderCheck], ReaderCheck]:
    def inner(check: ReaderCheck) -> ReaderCheck:
        READERS.append((name, check))
        return check
    return inner

def find_reader(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    for name, reader in READERS:
        try:
            source = reader(path, settings)
        except ImportError:
            logging.debug(f'Unable to check for {name} format - some dependencies may not be installed')
            continue
        if not source:
            logging.debug(f'File appears not to be [blue]{name}[/blue] format - skipping', extra={'markup': True})
            continue
        logging.info(f'File appears to be [blue]{name}[/blue] format', extra={'markup': True})
        return source
    return None


@register_reader('IFEM Eigenmodes')
def ifem_modes(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .ifem import IfemModes
    if IfemModes.applicable(path):
        return IfemModes(path)
    return None


@register_reader('IFEM')
def ifem(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .ifem import Ifem
    if Ifem.applicable(path):
        return Ifem(path)
    return None


@register_reader('SIMRA Map')
def simra_map(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import SimraMap
    if SimraMap.applicable(path):
        return SimraMap(path)
    return None


@register_reader('SIMRA 2D Mesh')
def simra_2dmesh(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import Simra2dMesh
    if Simra2dMesh.applicable(path):
        return Simra2dMesh(path)
    return None


@register_reader('SIMRA 3D Mesh')
def simra_3dmesh(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import Simra3dMesh
    if Simra3dMesh.applicable(path, settings):
        return Simra3dMesh(path)
    return None


@register_reader('SIMRA Boundary')
def simra_boundary(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import SimraBoundary
    if SimraBoundary.applicable(path, settings):
        return SimraBoundary(path)
    return None


@register_reader('SIMRA Continuation')
def simra_cont(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import SimraContinuation
    if SimraContinuation.applicable(path, settings):
        return SimraContinuation(path)
    return None


@register_reader('SIMRA History')
def simra_hist(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .simra import SimraHistory
    if SimraHistory.applicable(path, settings):
        return SimraHistory(path)
    return None


@register_reader('WRF')
def wrf(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    from .wrf import Wrf
    if Wrf.applicable(path):
        return Wrf(path)
    return None


@register_reader('GoTools')
def gotools(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    if path.suffix.lower() != '.g2':
        return None
    from .gotools import GoTools
    return GoTools(path)


@register_reader('LRSpline')
def lrspline(path: Path, settings: FindReaderSettings) -> Optional[Source]:
    if path.suffix.lower() != '.lr':
        return None
    from .lrspline import LrSpline
    return LrSpline(path)
