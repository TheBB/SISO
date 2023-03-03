# import numpy as np

# from .field_data import FieldData


# def spherical_cartesian_vector_field(data: FieldData, coords: FieldData, invert: bool = False) -> FieldData:
#     lon, lat, *_ = coords.components
#     lon = np.deg2rad(lon)
#     lat = np.deg2rad(lat)
#     clon, clat = np.cos(lon), np.cos(lat)
#     slon, slat = np.sin(lon), np.sin(lat)

#     retval = np.zeros_like(data.data)
#     u, v, w = data.components
#     retval[..., 0] -= slon * u
#     retval[..., 1] -= slat * slon * v
#     retval[..., 2] += slat * w

#     if invert:
#         retval[..., 1] -= slat * clon * u
#         retval[..., 2] += clat * clon * u
#         retval[..., 0] += clon * v
#         retval[..., 2] += clat * slon * v
#         retval[..., 1] += clat * w
#     else:
#         retval[..., 0] -= slat * clon * v
#         retval[..., 0] += clat * clon * w
#         retval[..., 1] += clon * u
#         retval[..., 1] += clat * slon * w
#         retval[..., 2] += clat * v

#     return FieldData(retval)
