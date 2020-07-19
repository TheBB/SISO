from os.path import splitext, basename
from pathlib import Path
import pytest
import re

import numpy as np
from click.testing import CliRunner

from .shared import TESTDATA_DIR, FILES, cd_temp
from ifem_to_vt.__main__ import convert

try:
    import vtfwriter
    has_vtf = True
except ImportError:
    has_vtf = False


NUM = re.compile(r'-?\d+[ +\-e.\d]*\n?$')


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname, _ = request.param
    base, _ = splitext(basename(rootname))
    vtfname = '{}.vtf'.format(base)
    return (
        TESTDATA_DIR / rootdir / rootname,
        TESTDATA_DIR / 'vtf' / vtfname,
        vtfname,
    )


def compare_vtf(out, ref):
    outiter = iter(out)
    refiter = iter(ref)
    for outline, refline in zip(outiter, refiter):
        if 'EXPORT_DATE' in outline:
            continue
        if NUM.match(refline):
            out = list(map(float, outline.split()))
            ref = list(map(float, refline.split()))
            np.testing.assert_allclose(out, ref, atol=1e-10)
        else:
            assert outline == refline

    # Check that files are equally long
    assert not list(outiter)
    assert not list(refiter)


@pytest.mark.skipif(not has_vtf, reason="VTF tests not runnable without vtfwriter")
def test_vtf_integrity(filenames):
    infile, checkfile, outfile = filenames
    with cd_temp() as tempdir:
        outfile = tempdir / outfile
        res = CliRunner().invoke(convert, ['--mode', 'ascii', '-f', 'vtf', str(infile)])
        assert res.exit_code == 0
        assert outfile.exists()
        assert checkfile.exists()
        with open(outfile, 'r') as out, open(checkfile, 'r') as ref:
            compare_vtf(out, ref)
