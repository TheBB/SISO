from os.path import splitext, basename
from pathlib import Path
import pytest

from click.testing import CliRunner

from .shared import TESTCASES, compare_vtk_unstructured, cd_temp
from ifem_to_vt.__main__ import convert

import vtk
has_vtk_9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


def load_grid(filename: Path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vtk(out: Path, ref: Path):
    assert out.exists()
    assert ref.exists()
    compare_vtk_unstructured(load_grid(str(out)), load_grid(ref))


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
@pytest.mark.parametrize('case', TESTCASES['vtk'])
def test_vtk_integrity(case):
    with cd_temp() as tempdir:
        res = CliRunner().invoke(convert, ['--mode', 'ascii', '-f', 'vtk', str(case.sourcefile)])
        assert res.exit_code == 0
        for out, ref in case.check_files(tempdir):
            compare_vtk(out, ref)
