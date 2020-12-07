import numpy as np

from ..typing import Array

try:
    import jax.numpy as jnp
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False



def spherical_cartesian_vf(lon: Array, lat: Array, data: Array, invert: bool = False) -> Array:
    """Convert a spherical vector field to a Cartesian vector field or back. """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    clon, clat = np.cos(lon), np.cos(lat)
    slon, slat = np.sin(lon), np.sin(lat)

    retval = np.zeros_like(data)
    retval[..., 0] -= slon * data[..., 0]
    retval[..., 1] -= slat * slon * data[..., 1]
    retval[..., 2] += slat * data[..., 2]

    if invert:
        retval[..., 1] -= slat * clon * data[..., 0]
        retval[..., 2] += clat * clon * data[..., 0]
        retval[..., 0] += clon * data[..., 1]
        retval[..., 2] += clat * slon * data[..., 1]
        retval[..., 1] += clat * data[..., 2]
    else:
        retval[..., 0] -= slat * clon * data[..., 1]
        retval[..., 0] += clat * clon * data[..., 2]
        retval[..., 1] += clon * data[..., 0]
        retval[..., 1] += clat * slon * data[..., 2]
        retval[..., 2] += clat * data[..., 1]

    return retval



# Universal Transversal Mercator
# The following code is modified from the UTM package by Tobies Bienek
# It has been made to work with Jax for auto-differentiation purposes,
# and it is licensed under the MIT License.
# ----------------------------------------------------------------------


K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = np.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137


def mod_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi

def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3

def normalize_pair(x, y):
    norm = np.sqrt(x**2 + y**2)
    return x / norm, y / norm

def utm_to_lonlat_vf(x: Array, y: Array, vx: Array, vy: Array, zone: int, zone_letter: str) -> Array:
    if not HAS_JAX:
        raise TypeError(f"Jax must be installed for UTM vector field conversion")
    x, y = jnp.array(x), jnp.array(y)
    lon = lambda x, y: _utm_to_lonlat(x, y, zone, zone_letter)[0]
    lat = lambda x, y: _utm_to_lonlat(x, y, zone, zone_letter)[1]
    dlon_dx, dlon_dy = jax.vmap(jax.grad(lon, (0, 1)))(x, y)
    dlat_dx, dlat_dy = jax.vmap(jax.grad(lat, (0, 1)))(x, y)
    dlon_dx, dlon_dy = normalize_pair(dlon_dx, dlon_dy)
    dlat_dx, dlat_dy = normalize_pair(dlat_dx, dlat_dy)
    return dlon_dx * vx + dlon_dy * vy, dlat_dx * vx + dlat_dy * vy

def utm_to_lonlat(x: Array, y: Array, zone: int, zone_letter: str) -> Array:
    return _utm_to_lonlat(x, y, zone, zone_letter)

def lonlat_to_utm(x: Array, y: Array, zone: int, zone_letter: str) -> Array:
    return _lonlat_to_utm(x, y, zone, zone_letter)

def _utm_to_lonlat(x: 'jnp.array', y: 'jnp.array', zone: int, zone_letter: str) -> 'jnp.array':
    zone_letter = zone_letter.upper()
    northern = (zone_letter >= 'N')

    x = x - 500000

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * jnp.sin(2 * mu) +
             P3 * jnp.sin(4 * mu) +
             P4 * jnp.sin(6 * mu) +
             P5 * jnp.sin(8 * mu))

    p_sin = jnp.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = jnp.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = jnp.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = E_P2 * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                 d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    longitude = mod_angle(longitude + jnp.radians(zone_number_to_central_longitude(zone)))

    return jnp.degrees(longitude), jnp.degrees(latitude)


def _lonlat_to_utm(lon: 'jnp.array', lat: 'jnp.array', zone: int, zone_letter: str) -> 'jnp.array':
    zone_letter = zone_letter.upper()

    lat_rad = jnp.radians(lat)
    lat_sin = jnp.sin(lat_rad)
    lat_cos = jnp.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    lon_rad = jnp.radians(lon)
    central_lon = (zone - 1) * 6 - 180 + 3
    central_lon_rad = jnp.radians(central_lon)

    n = R / jnp.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * jnp.sin(2 * lat_rad) +
             M3 * jnp.sin(4 * lat_rad) -
             M4 * jnp.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if jnp.max(lat) < 0:
        northing += 10000000

    return easting, northing
