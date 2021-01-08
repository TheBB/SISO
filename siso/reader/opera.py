from datetime import datetime
from itertools import count
from operator import itemgetter
from pathlib import Path

from typing import Iterable, Tuple, Union, Dict, List

import h5py
import numpy as np
import treelog as log

from .reader import Reader
from ..fields import Field, SimpleField, FieldPatches, Geometry, PatchData, FieldData
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

    def patch(self, coords: UTM) -> Tuple[PatchData, FieldData]:
        # Convert center
        c_lon, c_lat = self.where['lon'], self.where['lat']
        center = np.array(lonlat_to_utm(c_lon, c_lat, coords.zone_number, coords.zone_letter))

        # Convert a point to the north, to find zero azimuth
        north = np.array(lonlat_to_utm(c_lon, c_lat + 1e-3, coords.zone_number, coords.zone_letter)) - center
        north /= np.linalg.norm(north)
        zero_az = np.arctan2(north[1], north[0])

        anglesync = self.how.get('anglesync', b'azimuth')
        if anglesync == b'azimuth':
            elangle = self.where['elangle']

            # Compute radial and azimuthal space
            h_rscale = self.where['rscale'] * np.cos(np.deg2rad(elangle))
            rad = self.where['rstart'] + np.arange(self.where['nbins'] + 1) * h_rscale
            az = zero_az - np.deg2rad(self.how['startazA'])
            az = np.append(az, [az[0]])

            # Convert to cartesian coordinates
            rad, az = np.meshgrid(rad, az)
            x = rad * np.cos(az) + center[0]
            y = rad * np.sin(az) + center[1]

            # Compute altitude
            z = rad * np.sin(np.deg2rad(elangle)) + self.where['height']

        elif anglesync == b'elevation':
            # Compute radial and elevation space
            rad = self.where['rstart'] + np.arange(self.where['nbins'] + 1) * self.where['rscale']
            start_elev = self.how['startelA']
            stop_elev = self.how['stopelA']
            elev = np.deg2rad((np.append(start_elev, [stop_elev[-1]]) + np.append([start_elev[0]], stop_elev)) / 2)

            # Convert to cartesian coordinates
            rot = zero_az - np.deg2rad(self.where['azangle'])
            rad, elev = np.meshgrid(rad, elev)
            x = rad * np.cos(elev)
            y = x * np.sin(rot)
            x *= np.cos(rot)
            z = rad * np.sin(elev) + self.where['height']

        topo = StructuredTopology((self.where['nrays'], self.where['nbins']), celltype=Quad())
        nodes = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        return Patch(('geometry',), topo), nodes


class OperaField(SimpleField):

    data: List[OperaData]

    ncomps = 1
    cells = True
    decompose = False

    def __init__(self, name: str, data: List[OperaData]):
        self.name = name
        self.data = data

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
        yield dataset.patch(self.coords)


class OperaVolumeGeometryField(SimpleField):

    datasets: List[OperaDataset]

    name = 'Geometry'
    ncomps = 3
    cells = False
    decompose = False

    def __init__(self, datasets: List[OperaDataset]):
        assert all(dataset.what['product'] == b'SCAN' for dataset in datasets)
        self.datasets = datasets
        self.fieldtype = Geometry(UTM.optimal(datasets[0].where['lon'], datasets[0].where['lat']))

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        patches, nodes = zip(*(dataset.patch(self.coords) for dataset in self.datasets))
        topos = [patch.topology for patch in patches]
        newtopo = StructuredTopology.stack(*topos)
        newnodes = np.stack([n.reshape(*topos[0].nodeshape, -1) for n in nodes], axis=-2).reshape(-1, nodes[0].shape[-1])
        yield Patch(('geometry',), newtopo), newnodes
        # yield Patch(('geometry',), topos[0]), nodes[0]


class OperaVolumeField(SimpleField):

    data: List[OperaData]

    ncomps = 1
    cells = True
    decompose = False

    def __init__(self, name: str, data: List[OperaData]):
        self.name = name
        self.data = data

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        datasets = [OperaDataset(data.grp.parent) for data in self.data]
        patch, _ = next(OperaVolumeGeometryField(datasets).patches(stepid, force))

        datas = [d.data() for d in self.data]
        datas = [(a+b)/2 for a, b in zip(datas[:-1], datas[1:])]
        data = np.stack(datas, axis=-1).reshape(-1, 1)
        yield patch, data


class OperaReader(Reader):

    reader_name = "EUMETNET-OPERA"

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


class OperaPvolReader(OperaReader):

    object_type = 'PVOL'
    reader_name = "EUMETNET-OPERA-PVOL"

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        """Iterate over all steps with associated data."""
        yield (0, {'time': 0.0})

    def fields(self) -> Iterable[Field]:
        """Iterate over all fields."""

        datasets = {dataset.where['elangle']: dataset for dataset in self.datasets('SCAN')}
        datasets_sorted = list(map(itemgetter(1), sorted(datasets.items(), key=itemgetter(0))))

        field_data = {}
        for dataset in datasets_sorted:
            for data in dataset.data():
                name = data.what['quantity'].decode()
                field_data.setdefault(name, []).append(data)

        for name, datas in field_data.items():
            yield OperaVolumeField(name, datas)

        yield OperaVolumeGeometryField(datasets_sorted)
