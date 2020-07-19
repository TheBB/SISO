from contextlib import contextmanager
from os.path import join, dirname, splitext
from os import chdir
from pathlib import Path
import tempfile

import numpy as np

import vtk
import vtk.util.numpy_support as vtknp


@contextmanager
def cd_temp():
    with tempfile.TemporaryDirectory() as tempdir_path:
        tempdir = Path(tempdir_path)
        olddir = Path.cwd()
        try:
            chdir(tempdir)
            yield tempdir
        finally:
            chdir(olddir)


TESTDATA_DIR = Path(__file__).parent / 'testdata'

FILES = [
    ('hdf5', 'Annulus.hdf5', 3),
    ('hdf5', 'Cavity-mixed.hdf5', 1),
    ('hdf5', 'Cavity3D-compatible.hdf5', 1),
    ('hdf5', 'Cyl2D-VMSFSI-weak.hdf5', 11),
    ('hdf5', 'singular-pressure-corner-rec.hdf5', 3),
    ('hdf5', 'SmallBox.hdf5', 3),
    ('hdf5', 'Square.hdf5', 1),
    ('hdf5', 'Square-ad.hdf5', 11),
    ('hdf5', 'Square-LR.hdf5', 1),
    ('hdf5', 'Square-compatible-abd1-B-I-stat.hdf5', 1),
    ('hdf5', 'Square-mixed-abd1-B-I-stat.hdf5', 1),
    ('hdf5', 'Square-modes.hdf5', 10),
    ('hdf5', 'Square-modes-freq.hdf5', 10),
    ('hdf5', 'Waterfall3D.hdf5', 1),
    ('g2', 'Backstep2D.g2', None),
    ('g2', 'annulus3D.g2', None),
    ('lr', 'square-2.lr', None),
    ('lr', 'backstep-3.lr', None),
    ('lr', 'cube-3.lr', None),
    ('res', 'box/box.res', None),
]


def step_filenames(n, base):
    basename, ext = splitext(base)
    for i in range(1, n+1):
        yield Path('{}-{}{}'.format(basename, i, ext))


def compare_data(out, ref):
    assert out.GetNumberOfArrays() == ref.GetNumberOfArrays()
    narrays = out.GetNumberOfArrays()
    for i in range(narrays):
        assert out.GetArrayName(i) == ref.GetArrayName(i)
        np.testing.assert_allclose(
            vtknp.vtk_to_numpy(out.GetArray(i)),
            vtknp.vtk_to_numpy(ref.GetArray(i)),
            atol=1e-15,
        )


def compare_vtk_unstructured(out, ref):
    np.testing.assert_allclose(
        vtknp.vtk_to_numpy(out.GetPoints().GetData()),
        vtknp.vtk_to_numpy(ref.GetPoints().GetData()),
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        vtknp.vtk_to_numpy(out.GetCells().GetData()),
        vtknp.vtk_to_numpy(ref.GetCells().GetData()),
    )
    compare_data(out.GetPointData(), ref.GetPointData())
    compare_data(out.GetCellData(), ref.GetCellData())
