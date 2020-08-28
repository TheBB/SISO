from abc import ABC
from inspect import isabstract

from typing import Union

from .util import subclasses



# Reference ellipsoids
# ----------------------------------------------------------------------


class Ellipsoid:

    name: str
    erfa_code: int

    @staticmethod
    def find(name: str):
        for cls in subclasses(Ellipsoid, root=False):
            if cls.name == name.upper():
                return cls()
        raise ValueError(f"Unknown reference ellipsoid: {name.upper()}")

    def __str__(self):
        return self.name


class WGS84(Ellipsoid):

    name = 'WGS84'
    erfa_code = 1


class GRS80(Ellipsoid):

    name = 'GRS80'
    erfa_code = 2


class WGS72(Ellipsoid):

    name = 'WGS72'
    erfa_code = 3



# Coordinate systems
# ----------------------------------------------------------------------


class CoordinateSystem(ABC):

    name: str

    @staticmethod
    def find(name: str) -> 'CoordinateSystem':
        root, *args = name.lower().split(':')
        for cls in subclasses(CoordinateSystem, root=False, invert=True):
            if cls.name == root:
                return cls(*args)
        return Local(name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, CoordinateSystem):
            return False
        return str(self) == str(other)


class Local(CoordinateSystem):
    """This class represents an unspecified coordinate system to and
    from which conversion is impossible.  There may be several such,
    which are distinguished by name.
    """

    name = 'local'

    specific_name: str

    def __init__(self, name='local'):
        self.specific_name = name

    def __str__(self):
        return self.specific_name


class Geodetic(CoordinateSystem):
    """Latitude, longitude and height above the reference ellipsoid."""

    name = 'geodetic'


class Geocentric(CoordinateSystem):
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
