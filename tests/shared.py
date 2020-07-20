from contextlib import contextmanager
from os import chdir
from pathlib import Path
import tempfile

from typing import List, Optional

import numpy as np

import vtk
import vtk.util.numpy_support as vtknp


@contextmanager
def cd_temp():
    """Context manager that creates a temporary directory and sets cwd to
    it, restoring it after.
    """
    with tempfile.TemporaryDirectory() as tempdir_path:
        tempdir = Path(tempdir_path)
        olddir = Path.cwd()
        try:
            chdir(tempdir)
            yield tempdir
        finally:
            chdir(olddir)


class TestCase:
    """Data class representing a single test case, with source file,
    indended target format, and a list of output files as reference.
    """

    sourcefile: Path
    target_format: str
    reference_path: Path
    reference_files: List[str]

    def __init__(self, sourcefile, target_format, reference_path, reference_files):
        self.sourcefile = sourcefile
        self.target_format = target_format
        self.reference_path = reference_path
        self.reference_files = list(reference_files)
        assert len(self.reference_files) > 0

    def check_files(self, path):
        """Yield a sequence of tuples, the first being the output of the test
        and the second being the reference to check against.  The test
        output files should be found in the given path, usually the
        CWD of the running process.
        """
        for filename in self.reference_files:
            yield (path / filename, self.reference_path / filename)


# In the following we build a catalogue of test cases
# programmatically. The TESTCASES dictionary maps output format name
# to a list of test cases.
TESTCASES = {}

# The root of the repository's test data path.
TESTDATA_DIR = Path(__file__).parent / 'testdata'

# These formats support multiple time steps in one file. This is
# needed to know which files to compare after conversion.
MULTISTEP_FORMATS = {'pvd', 'vtf'}


def testcase(sourcefile, nsteps, *formats):
    """Create test cases for converting SOURCEFILE to every format listed
    in FORMATS.  NSTEPS should be None if the source data has no
    timesteps, or the number of steps if it does.
    """
    sourcefile = TESTDATA_DIR / sourcefile
    for fmt in formats:
        basename = Path(f'{sourcefile.stem}.{fmt}')
        TESTCASES.setdefault(fmt, []).append(TestCase(
            sourcefile, fmt, TESTDATA_DIR / fmt, filename_maker(fmt, fmt in MULTISTEP_FORMATS)(basename, nsteps)
        ))


def filename_maker(ext: Optional[str], multistep: bool):
    """Return a function that creates correct filenames for output
    formats.  EXT should be the expected extension (possibly None) and
    MULTISTEP should be True if the format supports multiple timesteps
    in one file.

    The returned function accepts two arguments: a filename and an
    optional integer for number of timesteps, and returns an iterator
    over the relevant filenames.
    """
    ext = f'.{ext}' if ext is not None else ''
    def maker(base: Path, nsteps: Optional[int] = None):
        if multistep or nsteps is None:
            yield Path(f'{base.stem}{ext}')
            return
        for i in range(1, nsteps + 1):
            yield Path(f'{base.stem}-{i}{ext}')
    return maker


# List of test cases
FORMATS = ['vtk', 'vtu', 'pvd', 'vtf']
testcase('hdf5/Annulus.hdf5', 3, *FORMATS)
testcase('hdf5/Cavity-mixed.hdf5', 1, *FORMATS)
testcase('hdf5/Cavity3D-compatible.hdf5', 1, *FORMATS)
testcase('hdf5/Cyl2D-VMSFSI-weak.hdf5', 11, *FORMATS)
testcase('hdf5/singular-pressure-corner-rec.hdf5', 3, *FORMATS)
testcase('hdf5/SmallBox.hdf5', 3, *FORMATS)
testcase('hdf5/Square.hdf5', 1, *FORMATS)
testcase('hdf5/Square-ad.hdf5', 11, *FORMATS)
testcase('hdf5/Square-LR.hdf5', 1, *FORMATS)
testcase('hdf5/Square-compatible-abd1-B-I-stat.hdf5', 1, *FORMATS)
testcase('hdf5/Square-mixed-abd1-B-I-stat.hdf5', 1, *FORMATS)
testcase('hdf5/Square-modes.hdf5', 10, *FORMATS)
testcase('hdf5/Square-modes-freq.hdf5', 10, *FORMATS)
testcase('hdf5/Waterfall3D.hdf5', 1, *FORMATS)
testcase('g2/Backstep2D.g2', None, *FORMATS)
testcase('g2/annulus3D.g2', None, *FORMATS)
testcase('lr/square-2.lr', None, *FORMATS)
testcase('lr/backstep-3.lr', None, *FORMATS)
testcase('lr/cube-3.lr', None, *FORMATS)
testcase('res/box/box.res', None, *FORMATS)


def compare_vtk_data(out, ref):
    """Helper function for comparing two vtkDataSetAttributes objects,
    generally vtkPointData or vtkCellData.
    """
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
    """Helper function for comparing two vtkDataSet objects."""
    np.testing.assert_allclose(
        vtknp.vtk_to_numpy(out.GetPoints().GetData()),
        vtknp.vtk_to_numpy(ref.GetPoints().GetData()),
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        vtknp.vtk_to_numpy(out.GetCells().GetData()),
        vtknp.vtk_to_numpy(ref.GetCells().GetData()),
    )
    compare_vtk_data(out.GetPointData(), ref.GetPointData())
    compare_vtk_data(out.GetCellData(), ref.GetCellData())
