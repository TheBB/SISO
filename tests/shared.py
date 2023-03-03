from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from os import chdir
from pathlib import Path
import tempfile

from click.testing import CliRunner
import numpy as np

import vtkmodules.util.numpy_support as vtknp

from siso.__main__ import main

from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)


@contextmanager
def cd_temp() -> Path:
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


@dataclass
class PreparedTestCase:
    """Data class representing a single test case, with source file,
    indended target format, and a list of output files as reference.
    """

    sourcefile: Path
    outfile: Path
    target_format: str
    reference_path: Path
    reference_files: List[Path]
    extra_args: List[str]
    abs_tol: float
    rel_tol: float

    def check_files(self, path: Path) -> Iterator[Tuple[Path, Path]]:
        """Yield a sequence of tuples, the first being the output of the test
        and the second being the reference to check against.  The test
        output files should be found in the given path, usually the
        CWD of the running process.
        """
        for filename in self.reference_files:
            yield (path / filename, self.reference_path / filename)

    @contextmanager
    def invoke(self, fmt: str, mode: str = 'ascii') -> Path:
        args = [
            '--debug',
            '--verify-strict',
            '--mode', mode,
            '-f', fmt,
            *self.extra_args,
            str(self.sourcefile),
            '-o', str(self.outfile)
        ]
        print(args)
        with cd_temp() as tempdir:
            res = CliRunner().invoke(main, args)
            if res.exit_code != 0:
                print(res.stdout)
                assert False
            yield tempdir


# In the following we build a catalogue of test cases
# programmatically. The TESTCASES dictionary maps output format name
# to a list of test cases.
TESTCASES: Dict[str, List[PreparedTestCase]] = {}
TESTIDS: Dict[str, List[str]] = {}

# The root of the repository's test data path.
TESTDATA_DIR = Path(__file__).parent / 'testdata'

# These formats support multiple time steps in one file. This is
# needed to know which files to compare after conversion.
MULTISTEP_FORMATS = {'pvd', 'vtf'}


def testcase(
    sourcefile: Path,
    nsteps: Optional[int],
    formats: Sequence[str],
    *extra_args: str,
    suffix: str = '',
    abs_tol: float = 2e-7,
    rel_tol: float  = 2e-7,
    format_args: Dict[str, List[str]] = {}
):
    """Create test cases for converting SOURCEFILE to every format listed
    in FORMATS.  NSTEPS should be None if the source data has no
    timesteps, or the number of steps if it does.
    """
    sourcefile = TESTDATA_DIR / sourcefile
    for fmt in formats:
        basename = Path(f'{sourcefile.stem}{suffix}.{fmt}')
        reference_files = filename_maker(fmt, fmt in MULTISTEP_FORMATS)(basename, nsteps)

        TESTCASES.setdefault(fmt, []).append(PreparedTestCase(
            sourcefile=sourcefile,
            outfile=basename,
            target_format=fmt,
            reference_path=TESTDATA_DIR / fmt,
            reference_files=reference_files,
            extra_args=list(extra_args) + format_args.get(fmt, []),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ))

        args = ' '.join(extra_args)
        TESTIDS.setdefault(fmt, []).append(f'{sourcefile.name} {args}')


def filename_maker(ext: Optional[str], multistep: bool) -> Iterator[Path]:
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
            return [Path(f'{base.stem}{ext}')]
        return [Path(f'{base.stem}-{i}{ext}') for i in range(1, nsteps + 1)]
    return maker


# These data sets are structured. The VTK reference files were
# generated before structured output was added, so they are compared
# using the --unstructured option.
formats = ['vtk', 'vtu', 'vts', 'pvd', 'vtf']
kwargs = {'format_args': {'vtk': ['--unstructured']}}
testcase('hdf5/Annulus.hdf5', 3, formats, **kwargs)
testcase('hdf5/Cavity-mixed.hdf5', 1, formats, **kwargs)
testcase('hdf5/Cavity3D-compatible.hdf5', 1, formats, **kwargs)
testcase('hdf5/Square.hdf5', 1, formats, **kwargs)
testcase('hdf5/Square-ad.hdf5', 11, formats, **kwargs)
testcase('hdf5/Square-compatible-abd1-B-I-stat.hdf5', 1, formats, **kwargs)
testcase('hdf5/Square-mixed-abd1-B-I-stat.hdf5', 1, formats, **kwargs)
testcase('hdf5/Square-modes.hdf5', 10, formats, '--ead', **kwargs)
testcase('hdf5/Square-modes-freq.hdf5', 10, formats, '--ead', **kwargs)
testcase('hdf5/Waterfall3D.hdf5', 1, formats, **kwargs)
testcase('g2/annulus3D.g2', None, formats, **kwargs)

# Single precision, therefore inflated tolerance
testcase('simra/box/box.res', None, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)
testcase('simra/box/map.dat', None, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)
testcase('simra/box/mesh2d.dat', None, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)
testcase('simra/box/mesh.dat', None, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)
testcase('simra/boun/boun.dat', None, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)
testcase('simra/hist/hist.res', 1, formats, **kwargs, abs_tol=1e-5, rel_tol=1e-5)

# Unstructured data sets
formats = ['vtk', 'vtu', 'pvd', 'vtf']
testcase('hdf5/Cyl2D-VMSFSI-weak.hdf5', 11, formats)
testcase('hdf5/NACA0015_a6_small_weak_mixed_SA.hdf5', 4, formats)
testcase('hdf5/singular-pressure-corner-rec.hdf5', 3, formats)
testcase('hdf5/Square-LR.hdf5', 1, formats)
testcase('hdf5/SmallBox.hdf5', 3, formats)
testcase('g2/Backstep2D.g2', None, formats)
testcase('lr/square-2.lr', None, formats)
testcase('lr/backstep-3.lr', None, formats)
testcase('lr/cube-3.lr', None, formats)

# 1D so far untested with VTF
formats = ['vtk', 'vtu', 'pvd']
testcase('hdf5/TestCell1D.hdf5', 1, formats)

# WRF reader with various options.  We only use PVD, VTU and VTS here
# to save some space.
pr = '--periodic'
pl = '--planar'
ex = '--extrude'
gl = ['--coords', 'geocentric']
for n in ['eastward', 'northward', 'outward']:
    formats = ['vtu', 'vts', 'pvd']
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, suffix='-volumetric')
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, pl, suffix='-planar')
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, ex, suffix='-extrude')

    # The geocentric reference cases were generated with a slightly different
    # coordinate transformation algorithm.  Instead of regenerating them, we
    # use elevated tolerances.
    formats = ['vtu', 'pvd']
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, *gl, suffix='-volumetric-global', rel_tol=2e-4, abs_tol=2e-6)
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, pl, *gl, suffix='-planar-global', rel_tol=2e-4, abs_tol=2e-6)
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, ex, *gl, suffix='-extrude-global', rel_tol=2e-4, abs_tol=2e-6)
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, pl, *gl, pr, suffix='-planar-periodic', rel_tol=2e-4, abs_tol=2e-6)
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, *gl, pr, suffix='-volumetric-periodic', rel_tol=2e-4, abs_tol=2e-6)
    testcase(f'wrf/wrfout_d01-{n}.nc', 4, formats, ex, *gl, pr, suffix='-extrude-periodic', rel_tol=2e-4, abs_tol=2e-6)

# Miscellaneous CLI options
formats = ['vtk', 'vtu', 'pvd', 'vtf']
kwargs = {'format_args': {'vtk': ['--unstructured']}}
# testcase('hdf5/SmallBox.hdf5', None, formats, '--last', suffix='-with-last')
# testcase('hdf5/Annulus.hdf5', 3, formats, '--nvis', '2', suffix='-with-nvis', **kwargs)
# testcase('g2/annulus3D.g2', None, formats, '--nvis', '5', suffix='-with-nvis', **kwargs)
# testcase('wrf/wrfout_d01-eastward.nc', 4, ['pvd'], '--no-fields', suffix='-no-fields')
# testcase('wrf/wrfout_d01-eastward.nc', 4, ['pvd'], '-l', 'U', '-l', 'V', '-l', 'W', suffix='-filtered')
# testcase('wrf/wrfout_d01-eastward.nc', 4, ['pvd'], '-l', 'U,V,W', suffix='-filtered')


def compare_vtk_data(out, ref, case: PreparedTestCase):
    """Helper function for comparing two vtkDataSetAttributes objects,
    generally vtkPointData or vtkCellData.
    """
    if out.GetNumberOfArrays() != ref.GetNumberOfArrays():
        out_arrays = {out.GetArrayName(i) for i in range(out.GetNumberOfArrays())}
        ref_arrays = {ref.GetArrayName(i) for i in range(ref.GetNumberOfArrays())}
        print(out_arrays - ref_arrays)
        print(ref_arrays - out_arrays)
        assert False
    narrays = out.GetNumberOfArrays()
    for i in range(narrays):
        name = ref.GetArrayName(i)
        print('Checking', name)
        np.testing.assert_allclose(
            vtknp.vtk_to_numpy(out.GetAbstractArray(name)),
            vtknp.vtk_to_numpy(ref.GetAbstractArray(i)),
            atol=case.abs_tol,
            rtol=case.rel_tol,
        )


def compare_vtk_unstructured(out, ref, case: PreparedTestCase):
    """Helper function for comparing two vtkDataSet objects."""
    print('Checking points')
    np.testing.assert_allclose(
        vtknp.vtk_to_numpy(out.GetPoints().GetData()),
        vtknp.vtk_to_numpy(ref.GetPoints().GetData()),
        atol=case.abs_tol,
        rtol=case.rel_tol,
    )
    np.testing.assert_array_equal(
        vtknp.vtk_to_numpy(out.GetCells().GetData()),
        vtknp.vtk_to_numpy(ref.GetCells().GetData()),
    )
    compare_vtk_data(out.GetPointData(), ref.GetPointData(), case)
    compare_vtk_data(out.GetCellData(), ref.GetCellData(), case)


def compare_vtk_structured(out, ref, case: PreparedTestCase):
    """Helper function for comparing two vtkDataSet objects."""
    assert out.GetDimensions() == ref.GetDimensions()
    print('Checking points')
    np.testing.assert_allclose(
        vtknp.vtk_to_numpy(out.GetPoints().GetData()),
        vtknp.vtk_to_numpy(ref.GetPoints().GetData()),
        atol=case.abs_tol,
        rtol=case.rel_tol,
    )
    compare_vtk_data(out.GetPointData(), ref.GetPointData(), case)
    compare_vtk_data(out.GetCellData(), ref.GetCellData(), case)
