from os.path import join, splitext, dirname, exists, basename
import pytest
import tempfile

from .shared import TESTDATA_DIR, FILES, step_filenames, compare_vtk_unstructured

from ifem_to_vt import config
from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer

import vtk
has_vtk_9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname = request.param
    base, _ = splitext(basename(rootname))
    vtuname = '{}.vtu'.format(base)
    return (
        join(TESTDATA_DIR, rootdir, rootname),
        join(TESTDATA_DIR, 'vtu', vtuname),
        vtuname,
    )


def load_grid(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def compare_vtu(out, ref):
    assert exists(out)
    assert exists(ref)
    compare_vtk_unstructured(load_grid(out), load_grid(ref))


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
def test_vtu_integrity(filenames):
    infile, checkfile, outfile = filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with config(mode='ascii'):
            with get_reader(infile) as r, get_writer('vtu')(outfile) as w:
                nsteps = getattr(r, 'nsteps', 0)
                r.write(w)
            if config.last:
                compare_vtu(outfile, checkfile)
            else:
                for outfn, checkfn in zip(step_filenames(nsteps, outfile), step_filenames(nsteps, checkfile)):
                    compare_vtu(outfn, checkfn)
