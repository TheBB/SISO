from datetime import datetime
from itertools import count
from pathlib import Path

from typing import Iterable, Tuple, Union, Dict, List

import h5py
import numpy as np

from .reader import Reader
from ..fields import Field, SimpleField, FieldPatches, Geometry
from ..geometry import StructuredTopology, Patch, Quad
from ..typing import StepData
from ..coords import UTM
from ..coords.util import lonlat_to_utm


def interesting_groups(grp: Union[h5py.Group, h5py.File]) -> Iterable[h5py.Group]:
    for grpname, subgrp in grp.items():
        if grpname in {'how', 'what', 'where'}:
            continue
        yield subgrp


def attribute_dicts(grp: Union[h5py.Group, h5py.File], name: str) -> Iterable[Dict]:
    if name in grp:
        yield grp[name].attrs
    if grp.name != '/':
        yield from attribute_dicts(grp.parent, name)


class OperaGroup:

    grp: h5py.Group

    def __init__(self, grp: h5py.Group):
        self.grp = grp

    def attributes(self, name: str) -> Dict:
        retval = dict()
        for d in attribute_dicts(self.grp, name):
            retval.update(d)
        return retval

    @property
    def what(self) -> Dict:
        return self.attributes('what')

    @property
    def where(self) -> Dict:
        return self.attributes('where')

    @property
    def how(self) -> Dict:
        return self.attributes('how')


class OperaData(OperaGroup):

    def data(self):
        return self.what['gain'] * self.grp['data'][:] + self.what['offset']


class OperaDataset(OperaGroup):

    def data(self):
        for subgrp in interesting_groups(self.grp):
            yield OperaData(subgrp)


class OperaField(SimpleField):

    data: List[OperaData]

    ncomps = 1
    cells = True
    decompose = False

    def __init__(self, name: str, data: List[OperaData]):
        self.name = name
        self.data = data
        self.ncomps = 1
        self.cells = True

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        data = self.data[stepid]
        dataset = OperaDataset(data.grp.parent)
        topo = StructuredTopology((dataset.where['nrays'], dataset.where['nbins']), celltype=Quad())
        yield Patch(('geometry',), topo), self.data[stepid].data().reshape(-1, 1)


class OperaGeometryField(SimpleField):

    dataset: OperaDataset

    name = 'Geometry'
    ncomps = 3
    cells = False
    decompose = False

    def __init__(self, dataset: OperaDataset):
        self.dataset = dataset
        assert dataset.what['product'] == b'SCAN'
        self.fieldtype = Geometry(UTM.optimal(dataset.where['lon'], dataset.where['lat']))

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        dataset = OperaDataset(self.dataset.grp.parent[f'dataset{stepid+1}'])
        elangle = dataset.where.get('elangle', 0.0)

        # Convert center
        c_lon, c_lat = dataset.where['lon'], dataset.where['lat']
        center = np.array(lonlat_to_utm(c_lon, c_lat, self.coords.zone_number, self.coords.zone_letter))

        # Convert a point to the north, to find zero azimuth
        north = np.array(lonlat_to_utm(c_lon, c_lat + 1e-3, self.coords.zone_number, self.coords.zone_letter)) - center
        north /= np.linalg.norm(north)

        if dataset.how['anglesync'] == b'azimuth':
            elangle = dataset.where['elangle']

            # Compute radial and azimuthal space
            h_rscale = dataset.where['rscale'] * np.cos(np.deg2rad(elangle))
            rad = dataset.where['rstart'] + np.arange(dataset.where['nbins'] + 1) * h_rscale
            az = np.pi / 2 - np.deg2rad(dataset.how['startazA'])
            az = np.append(az, [az[0]])

            # Convert to cartesian coordinates
            rad, az = np.meshgrid(rad, az)
            x = rad * np.cos(az) + center[0]
            y = rad * np.sin(az) + center[1]

            # Compute altitude
            z = rad * np.sin(np.deg2rad(elangle)) + dataset.where['height']

        elif dataset.how['anglesync'] == b'elevation':
            # Compute radial and elevation space
            rad = dataset.where['rstart'] + np.arange(dataset.where['nbins'] + 1) * dataset.where['rscale']
            start_elev = dataset.how['startelA']
            stop_elev = dataset.how['stopelA']
            elev = np.deg2rad((np.append(start_elev, [stop_elev[-1]]) + np.append([start_elev[0]], stop_elev)) / 2)

            # Convert to cartesian coordinates
            rot = np.pi / 2 - np.deg2rad(dataset.where['azangle'])
            rad, elev = np.meshgrid(rad, elev)
            x = rad * np.cos(elev)
            y = x * np.sin(rot)
            x *= np.cos(rot)
            z = rad * np.sin(elev) + dataset.where['height']

        topo = StructuredTopology((dataset.where['nrays'], dataset.where['nbins']), celltype=Quad())
        nodes = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        yield Patch(('geometry',), topo), nodes


class OperaReader(Reader):

    object_type: bytes

    filename: Path
    h5: h5py.File

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            with h5py.File(filename, 'r') as f:
                assert 'how' in f
                assert 'what' in f
                assert 'where' in f
                assert f['what'].attrs['object'].decode() == cls.object_type
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.h5 = h5py.File(self.filename, 'r').__enter__()
        print(dict(self.h5['what'].attrs))
        return self

    def __exit__(self, *args):
        return self.h5.__exit__(*args)

    def datasets(self, product: str) -> Iterable[OperaDataset]:
        for i in count(1):
            name = f'dataset{i}'
            if name in self.h5:
                dataset = OperaDataset(self.h5[name])
                if dataset.what['product'].decode() == product:
                    yield dataset
            else:
                break

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        """Iterate over all steps with associated data."""
        first_time = None
        for i, dataset in enumerate(self.datasets('SCAN')):
            time = datetime.strptime(dataset.what['startdate'].decode() + dataset.what['starttime'].decode(), '%Y%m%d%H%M%S')
            if first_time is None:
                first_time = time
                deltat = 0.0
            else:
                deltat = (time - first_time).total_seconds()
            yield (i, {'time': deltat})

    def fields(self) -> Iterable[Field]:
        """Iterate over all fields."""
        field_data = {}

        for dataset in self.datasets('SCAN'):
            for data in dataset.data():
                name = data.what['quantity'].decode()
                field_data.setdefault(name, []).append(data)

        for name, datas in field_data.items():
            yield OperaField(name, datas)

        yield OperaGeometryField(next(self.datasets('SCAN')))


class OperaScanReader(OperaReader):

    object_type = 'SCAN'
    reader_name = "EUMETNET-OPERA-SCAN"


class OperaElevReader(OperaReader):

    object_type = 'ELEV'
    reader_name = "EUMETNET-OPERA-ELEV"
