from abc import ABC, abstractmethod
from inspect import isabstract

import erfa
import numpy as np

from typing import Union, Dict, List, Tuple, Callable, Set, Iterable, Optional
from .typing import Array2D

from .util import subclasses, spherical_cartesian_vf
from . import config



# Errors
# ----------------------------------------------------------------------


class CoordinateConversionError(TypeError):
    pass



# Reference ellipsoids
# ----------------------------------------------------------------------


class Ellipsoid(ABC):

    name: str

    @staticmethod
    def find(name: str):
        for cls in subclasses(Ellipsoid, root=False):
            if hasattr(cls, 'name') and cls.name.lower() == name.lower():
                return cls()
        raise ValueError(f"Unknown reference ellipsoid: {name.lower()}")

    def __str__(self):
        return self.name

    @property
    @abstractmethod
    def parameters(self) -> Tuple[float, float]:
        """Get the semi-major axis and flattening."""


class SphericalEllipsoid(Ellipsoid):

    name = 'sphere'

    @property
    def parameters(self) -> Tuple[float, float]:
        mean_radius = 6_371_008.8
        return mean_radius, 0.0


class ERFAEllipsoid(Ellipsoid):

    erfa_code: int

    @property
    def parameters(self) -> Tuple[float, float]:
        return erfa.eform(self.erfa_code)


class WGS84(ERFAEllipsoid):

    name = 'WGS84'
    erfa_code = 1


class GRS80(ERFAEllipsoid):

    name = 'GRS80'
    erfa_code = 2


class WGS72(ERFAEllipsoid):

    name = 'WGS72'
    erfa_code = 3



# Coordinate systems
# ----------------------------------------------------------------------


class Coords(ABC):

    name: str

    @staticmethod
    def find(name: str) -> 'Coords':
        root, *args = name.lower().split(':')
        for cls in subclasses(Coords, root=False, invert=True):
            if cls.name == root:
                return cls(*args)
        return Local(name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Coords):
            return False
        return str(self) == str(other)

    def substitute(self):
        return self


class Local(Coords):
    """This class represents an unspecified coordinate system to and
    from which conversion is impossible.  There may be several such,
    which are distinguished by name.
    """

    name = 'local'

    specific_name: str

    def __init__(self, name='local'):
        self.specific_name = name

    def __str__(self):
        return f'Local({self.specific_name})'

    def substitute(self):
        return config.input_coords.get(self.specific_name, self)


class Geodetic(Coords):
    """Latitude, longitude and height above the reference ellipsoid."""

    name = 'geodetic'


class Geocentric(Coords):
    """Geocentric coordinates with origin in the center of the Earth,
    positive z pointing toward the north pole, the xy-plane aligned
    with the equator, and positive x pointing in the direction of the
    prime meridian.
    """

    ellipsoid: Ellipsoid

    name = 'geocentric'

    def __init__(self, ellipsoid: Union[Ellipsoid, str] = WGS84()):
        if isinstance(ellipsoid, str):
            ellipsoid = Ellipsoid.find(ellipsoid)
        self.ellipsoid = ellipsoid

    def __str__(self):
        return f'{self.name}:{self.ellipsoid}'



# Coordinate conversion
# ----------------------------------------------------------------------


ConvDict = Dict[Tuple[str, str], Callable]

class ConversionGraph:

    # Keeps track of neighboring coordinate systems: those that can be
    # directly reached by one converter
    neighbors: Dict[str, List[str]]

    # Keeps track of the actual point converter functions
    point_converters: ConvDict

    # Keeps track of the actual vector field converter functions
    vector_converters: ConvDict

    def __init__(self):
        self.neighbors = dict()
        self.point_converters = dict()
        self.vector_converters = dict()

    def points(self, source: str, target: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.neighbors.setdefault(source, []).append(target)
            self.point_converters[(source, target)] = func
            return func
        return decorator

    def vectors(self, source: str, target: str, trivial: bool = True) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.neighbors.setdefault(source, []).append(target)
            self.vector_converters[(source, target)] = func
            func.is_trivial = trivial
            return func
        return decorator

    def path(self, source: Coords, target: Coords) -> 'Converter':
        if source == target:
            return Converter(self, [])

        # Special case: conversion from Local(anything) to
        # Local('local') is always allowed and is a no-op
        if isinstance(source, Local) and target == Local('local'):
            return Converter(self, [])

        seen: Set[str] = set()
        front: Dict[str, List[str]] = {source.name: [source.name]}

        while front:
            new_front = dict()
            for front_point, path in front.items():
                for neighbor in self.neighbors.get(front_point, []):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    new_path = [*path, neighbor]
                    if neighbor == target.name:
                        return Converter(self, new_path)
                    new_front[neighbor] = new_path
            front = new_front

        raise CoordinateConversionError(f"Unable to convert {source} to {target}")

    def optimal_source(self, target: Coords, sources: Iterable[Coords]) -> Tuple[int, 'Converter']:
        min_distance, retval = None, None
        for i, source in enumerate(sources):
            try:
                converter = self.path(source, target)
            except ValueError:
                continue
            if min_distance is None or len(converter) < min_distance:
                min_distance = len(converter)
                retval = (i, converter)

        if retval is None:
            raise ValueError(f"Unable to find a conversion path to {target}")
        return retval

graph = ConversionGraph()


class Converter:

    graph: ConversionGraph
    path: List[str]

    def __init__(self, graph: ConversionGraph, path: List[str]):
        self.graph = graph
        self.path = path

    def __len__(self):
        return len(self.path)

    @property
    def is_trivial(self):
        return all(
            self.graph.vector_converters[(a, b)].is_trivial
            for a, b in zip(self.path[:-1], self.path[1:])
        )

    def convert(self, src: Coords, tgt: Coords, data: Array2D, lookup: ConvDict, **kwargs) -> Array2D:
        if not self.path:
            return data
        path = [src] + [Coords.find(c) for c in self.path[1:-1]] + [tgt]
        for a, b in zip(path[:-1], path[1:]):
            data = lookup[(a.name, b.name)](a, b, data, **kwargs)
        return data

    def points(self, src: Coords, tgt: Coords, data: Array2D) -> Array2D:
        return self.convert(src, tgt, data, self.graph.point_converters)

    def vectors(self, src: Coords, tgt: Coords, data: Array2D, nodes: Optional[Array2D] = None) -> Array2D:
        return self.convert(src, tgt, data, self.graph.vector_converters, nodes=nodes)


@graph.points('geodetic', 'geocentric')
def _(src: Geodetic, tgt: Geocentric, data: Array2D) -> Array2D:
    lon, lat, h = data.T
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    a, f = tgt.ellipsoid.parameters
    return erfa.gd2gce(a, f, lon, lat, h)

@graph.vectors('geodetic', 'geocentric', trivial=False)
def _(src: Geodetic, tgt: Geocentric, data: Array2D, nodes: Array2D) -> Array2D:
    lon, lat = nodes[:,0], nodes[:,1]
    return spherical_cartesian_vf(lon, lat, data)
