from os.path import splitext, basename
from pathlib import Path
import pytest

from click.testing import CliRunner

from .shared import TESTDATA_DIR, FILES, step_filenames, compare_vtk_unstructured, cd_temp
from ifem_to_vt.__main__ import convert

import vtk
has_vtk_9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname, nsteps = request.param
    base, _ = splitext(basename(rootname))
    vtuname = '{}.vtu'.format(base)
    return (
        TESTDATA_DIR / rootdir / rootname,
        TESTDATA_DIR / 'vtu' / vtuname,
        Path(vtuname),
        nsteps,
    )


def load_grid(filename: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vtu(out: Path, ref: Path):
    assert out.exists()
    assert ref.exists()
    compare_vtk_unstructured(load_grid(out), load_grid(ref))


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
def test_vtu_integrity(filenames):
    infile, checkfile, outfile, nsteps = filenames
    with cd_temp() as tempdir:
        outfile = tempdir / outfile
        res = CliRunner().invoke(convert, ['--mode', 'ascii', '-f', 'vtu', str(infile)])
        assert res.exit_code == 0
        if nsteps is None:
            compare_vtu(outfile, checkfile)
        else:
            for outfn, checkfn in zip(step_filenames(nsteps, outfile), step_filenames(nsteps, checkfile)):
                compare_vtu(outfn, checkfn)
