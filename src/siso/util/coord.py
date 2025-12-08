from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from attrs import define, field
from numpy.polynomial import Polynomial

if TYPE_CHECKING:
    from jax.typing import ArrayLike as JaxArray

    from siso.types import Floatd, FloatVector


def normalize_pair(x: JaxArray, y: JaxArray) -> tuple[JaxArray, JaxArray]:
    norm = np.sqrt(x**2 + y**2)
    return x / norm, y / norm


@define
class UtmConverter:
    semi_major_axis: float
    flattening: float
    zone: int
    northern: bool

    k0: ClassVar[float] = 0.9996
    E0: ClassVar[float] = 500_000.0

    ref_lon: float = field(init=False)
    n: float = field(init=False)
    A: float = field(init=False)
    alpha: tuple[float, float, float] = field(init=False)
    beta: tuple[float, float, float] = field(init=False)
    delta: tuple[float, float, float] = field(init=False)

    N0: float = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.ref_lon = np.deg2rad((self.zone - 1) * 6 - 180 + 3)
        self.N0 = 0.0 if self.northern else 10_000_000.0
        self.n = self.flattening / (2 - self.flattening)
        self.A = self.semi_major_axis / (1 + self.n) * (1 + self.n**2 / 4 + self.n**4 / 64)

        self.alpha = (
            cast("float", Polynomial([0, 1 / 2, -2 / 3, 5 / 16])(self.n)),
            cast("float", Polynomial([0, 0, 13 / 48, -3 / 5])(self.n)),
            cast("float", Polynomial([0, 0, 0, 61 / 240])(self.n)),
        )

        self.beta = (
            cast("float", Polynomial([0, 1 / 2, -2 / 3, 37 / 96])(self.n)),
            cast("float", Polynomial([0, 0, 1 / 48, 1 / 15])(self.n)),
            cast("float", Polynomial([0, 0, 0, 17 / 480])(self.n)),
        )

        self.delta = (
            cast("float", Polynomial([0, 2, -2 / 3, -2])(self.n)),
            cast("float", Polynomial([0, 0, 7 / 3, -8 / 5])(self.n)),
            cast("float", Polynomial([0, 0, 0, 56 / 15])(self.n)),
        )

    def to_utm(self, lon: FloatVector, lat: FloatVector) -> tuple[FloatVector, FloatVector]:
        e, n = self._to_utm(jnp.array(lon), jnp.array(lat))
        return (
            cast("FloatVector", np.array(e, dtype=lon.dtype)),
            cast("FloatVector", np.array(n, dtype=lon.dtype)),
        )

    def to_lonlat(self, e: FloatVector, n: FloatVector) -> tuple[FloatVector, FloatVector]:
        lon, lat = self._to_lonlat(jnp.array(e), jnp.array(n))
        return (
            cast("FloatVector", np.array(lon, dtype=e.dtype)),
            cast("FloatVector", np.array(lat, dtype=e.dtype)),
        )

    def to_utm_vf(
        self, lon: FloatVector, lat: FloatVector, vx: FloatVector, vy: FloatVector
    ) -> tuple[FloatVector, FloatVector]:
        lonj, latj = jnp.array(lon), jnp.array(lat)

        de_dlon, de_dlat = jax.vmap(jax.grad(self._to_easting, (0, 1)))(lonj, latj)
        de_dlon, de_dlat = normalize_pair(de_dlon, de_dlat)

        dn_dlon, dn_dlat = jax.vmap(jax.grad(self._to_northing, (0, 1)))(lonj, latj)
        dn_dlon, dn_dlat = normalize_pair(dn_dlon, dn_dlat)

        dtype: Floatd = lon.dtype
        return (
            np.array(de_dlon, dtype=dtype) * vx + np.array(de_dlat, dtype=dtype) * vy,  # type: ignore[operator]
            np.array(dn_dlon, dtype=dtype) * vx + np.array(dn_dlat, dtype=dtype) * vy,  # type: ignore[operator]
        )

    def to_lonlat_vf(
        self, e: FloatVector, n: FloatVector, vx: FloatVector, vy: FloatVector
    ) -> tuple[FloatVector, FloatVector]:
        ej, nj = jnp.array(e), jnp.array(n)

        dlon_dx, dlon_dy = jax.vmap(jax.grad(self._to_lon, (0, 1)))(ej, nj)
        dlon_dx, dlon_dy = normalize_pair(dlon_dx, dlon_dy)

        dlat_dx, dlat_dy = jax.vmap(jax.grad(self._to_lat, (0, 1)))(ej, nj)
        dlat_dx, dlat_dy = normalize_pair(dlat_dx, dlat_dy)

        dtype: Floatd = e.dtype
        return (
            np.array(dlon_dx, dtype=dtype) * vx + np.array(dlon_dy, dtype=dtype) * vy,  # type: ignore[operator]
            np.array(dlat_dx, dtype=dtype) * vx + np.array(dlat_dy, dtype=dtype) * vy,  # type: ignore[operator]
        )

    def _to_easting(self, lon: JaxArray, lat: JaxArray) -> JaxArray:
        easting, _ = self._to_utm(lon, lat)
        return easting

    def _to_northing(self, lon: JaxArray, lat: JaxArray) -> JaxArray:
        _, northing = self._to_utm(lon, lat)
        return northing

    def _to_utm(self, lon: JaxArray, lat: JaxArray) -> tuple[JaxArray, JaxArray]:
        lon = jnp.deg2rad(lon)
        lat = jnp.deg2rad(lat)

        u = 2 * np.sqrt(self.n) / (1 + self.n)
        sin_lat = jnp.sin(lat)

        t = jnp.sinh(jnp.arctanh(sin_lat) - u * jnp.arctanh(u * sin_lat))
        xi = jnp.arctan(t / jnp.cos(lon - self.ref_lon))
        eta = jnp.arctanh(jnp.sin(lon - self.ref_lon) / jnp.sqrt(1 + t**2))

        easting = self.E0 + self.k0 * self.A * (
            eta
            + self.alpha[0] * jnp.cos(2 * xi) * jnp.sinh(2 * eta)
            + self.alpha[1] * jnp.cos(4 * xi) * jnp.sinh(4 * eta)
            + self.alpha[2] * jnp.cos(6 * xi) * jnp.sinh(6 * eta)
        )

        northing = self.N0 + self.k0 * self.A * (
            xi
            + self.alpha[0] * jnp.sin(2 * xi) * jnp.cosh(2 * eta)
            + self.alpha[1] * jnp.sin(4 * xi) * jnp.cosh(4 * eta)
            + self.alpha[2] * jnp.sin(6 * xi) * jnp.cosh(6 * eta)
        )

        return easting, northing

    def _to_lon(self, easting: JaxArray, northing: JaxArray) -> JaxArray:
        lon, _ = self._to_lonlat(easting, northing)
        return lon

    def _to_lat(self, easting: JaxArray, northing: JaxArray) -> JaxArray:
        _, lat = self._to_lonlat(easting, northing)
        return lat

    def _to_lonlat(self, easting: JaxArray, northing: JaxArray) -> tuple[JaxArray, JaxArray]:
        xi = (northing - self.N0) / self.k0 / self.A
        eta = (easting - self.E0) / self.k0 / self.A

        xip = (
            xi
            - self.beta[0] * jnp.sin(2 * xi) * jnp.cosh(2 * eta)
            - self.beta[1] * jnp.sin(4 * xi) * jnp.cosh(4 * eta)
            - self.beta[2] * jnp.sin(6 * xi) * jnp.cosh(6 * eta)
        )

        etap = (
            eta
            - self.beta[0] * jnp.cos(2 * xi) * jnp.sinh(2 * eta)
            - self.beta[1] * jnp.cos(4 * xi) * jnp.sinh(4 * eta)
            - self.beta[2] * jnp.cos(6 * xi) * jnp.sinh(6 * eta)
        )

        chi = jnp.arcsin(jnp.sin(xip) / jnp.cosh(etap))

        lat = (
            chi
            + self.delta[0] * jnp.sin(2 * chi)
            + self.delta[1] * jnp.sin(4 * chi)
            + self.delta[2] * jnp.sin(6 * chi)
        )

        lon = self.ref_lon + jnp.arctan(jnp.sinh(etap) / jnp.cos(xip))

        return jnp.rad2deg(lon), jnp.rad2deg(lat)
