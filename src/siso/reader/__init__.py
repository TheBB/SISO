from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from attrs import define

from siso.api import Endianness, Source
from siso.util import Registry


@define
class FindReaderSettings:
    """Settings passed to the find_reader function.

    This is a subset of attributes from the `siso.api.ReaderSettings` class,
    whichever ones are deemed necessary to decide which reader to use.
    """

    endianness: Endianness
    mesh_filename: Path | None


# A callable that accepts a path and settings and determines whether this path
# can be opened with a given reader class, returning it if so.
ReaderCheck = Callable[[Path, FindReaderSettings], Source | None]


# Registry of reader functions.
readers: Registry[ReaderCheck] = Registry()


def find_reader(path: Path, settings: FindReaderSettings) -> Source | None:
    """Find a source object that can read the dataset at the given path with the
    given settings, if possible.
    """

    discarded: list[str] = []

    # Try each possible reader in turn
    for name, reader in readers.items():
        try:
            source = reader(path, settings)
        except ImportError:
            # Some readers may have optional dependencies which are only checked
            # at this point. They should manifest as import errors.
            logging.debug(f"Unable to check for {name} format - some dependencies may not be installed")
            continue
        if not source:
            discarded.append(name)
            continue
        logging.info(f"File appears to be [blue]{name}[/blue] format", extra={"markup": True})
        return source

    logging.error("Unable to find a suitable reader, tried these types:")
    for name in discarded:
        logging.error(f"- {name}")

    return None


# To properly handle optional dependencies, reader modules are only imported
# when the relevant function is called, not before!


@readers.register("IFEM Eigenmodes")
def ifem_modes(path: Path, settings: FindReaderSettings) -> Source | None:
    from .ifem import IfemModes

    if IfemModes.applicable(path):
        return IfemModes(path)
    return None


@readers.register("IFEM")
def ifem(path: Path, settings: FindReaderSettings) -> Source | None:
    from .ifem import Ifem

    if Ifem.applicable(path):
        return Ifem(path)
    return None


@readers.register("SIMRA Map")
def simra_map(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import SimraMap

    if SimraMap.applicable(path):
        return SimraMap(path)
    return None


@readers.register("SIMRA 2D Mesh")
def simra_2dmesh(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import Simra2dMesh

    if Simra2dMesh.applicable(path):
        return Simra2dMesh(path)
    return None


@readers.register("SIMRA 3D Mesh")
def simra_3dmesh(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import Simra3dMesh

    if Simra3dMesh.applicable(path, settings):
        return Simra3dMesh(path)
    return None


@readers.register("SIMRA Boundary")
def simra_boundary(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import SimraBoundary

    if SimraBoundary.applicable(path, settings):
        return SimraBoundary(path, settings)
    return None


@readers.register("SIMRA Continuation")
def simra_cont(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import SimraContinuation

    if SimraContinuation.applicable(path, settings):
        return SimraContinuation(path, settings)
    return None


@readers.register("SIMRA History")
def simra_hist(path: Path, settings: FindReaderSettings) -> Source | None:
    from .simra import SimraHistory

    if SimraHistory.applicable(path, settings):
        return SimraHistory(path, settings)
    return None


@readers.register("WRF")
def wrf(path: Path, settings: FindReaderSettings) -> Source | None:
    from .wrf import Wrf

    if Wrf.applicable(path):
        return Wrf(path)
    return None


@readers.register("GeoGrid")
def geogrid(path: Path, settings: FindReaderSettings) -> Source | None:
    from .wrf import GeoGrid

    if GeoGrid.applicable(path):
        return GeoGrid(path)
    return None


@readers.register("GoTools")
def gotools(path: Path, settings: FindReaderSettings) -> Source | None:
    if path.suffix.casefold() != ".g2":
        return None
    from .gotools import GoTools

    return GoTools(path)


@readers.register("LRSpline")
def lrspline(path: Path, settings: FindReaderSettings) -> Source | None:
    if path.suffix.casefold() != ".lr":
        return None
    from .lrspline import LrSpline

    return LrSpline(path)


@readers.register("Nastran")
def nastran(path: Path, settings: FindReaderSettings) -> Source | None:
    if path.suffix.casefold() not in (".nas", ".bdf"):
        return None
    from .nastran import Nastran

    return Nastran(path)
