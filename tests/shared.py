from os.path import join, dirname, splitext

import numpy as np

import vtk
import vtk.util.numpy_support as vtknp


TESTDATA_DIR = join(dirname(__file__), 'testdata')

FILES = [
    ('hdf5', 'Annulus.hdf5'),
    ('hdf5', 'Cavity-mixed.hdf5'),
    ('hdf5', 'Cavity3D-compatible.hdf5'),
    ('hdf5', 'Cyl2D-VMSFSI-weak.hdf5'),
    ('hdf5', 'singular-pressure-corner-rec.hdf5'),
    ('hdf5', 'Square.hdf5'),
    ('hdf5', 'Square-ad.hdf5'),
    ('hdf5', 'Square-LR.hdf5'),
    ('hdf5', 'Square-compatible-abd1-B-I-stat.hdf5'),
    ('hdf5', 'Square-mixed-abd1-B-I-stat.hdf5'),
    ('hdf5', 'Square-modes.hdf5'),
    ('hdf5', 'Square-modes-freq.hdf5'),
    ('hdf5', 'Waterfall3D.hdf5'),
    ('g2', 'Backstep2D.g2'),
    ('g2', 'annulus3D.g2'),
    ('lr', 'square-2.lr'),
    ('lr', 'backstep-3.lr'),
    ('res', 'box/box.res'),
]


def step_filenames(n, base):
    basename, ext = splitext(base)
    for i in range(1, n+1):
        yield '{}-{}{}'.format(basename, i, ext)


def compare_data(out, ref):
    assert out.GetNumberOfArrays() == ref.GetNumberOfArrays()
    narrays = out.GetNumberOfArrays()
    for i in range(narrays):
        assert out.GetArrayName(i) == ref.GetArrayName(i)
        np.testing.assert_allclose(
            vtknp.vtk_to_numpy(out.GetArray(i)),
            vtknp.vtk_to_numpy(ref.GetArray(i)),
        )


def compare_vtk_unstructured(out, ref):
    np.testing.assert_allclose(
        vtknp.vtk_to_numpy(out.GetPoints().GetData()),
        vtknp.vtk_to_numpy(ref.GetPoints().GetData()),
    )
    np.testing.assert_array_equal(
        vtknp.vtk_to_numpy(out.GetCells().GetData()),
        vtknp.vtk_to_numpy(ref.GetCells().GetData()),
    )
    compare_data(out.GetPointData(), ref.GetPointData())
    compare_data(out.GetCellData(), ref.GetCellData())
