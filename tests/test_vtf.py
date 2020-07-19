from os.path import join, splitext, exists, basename
import pytest
import re
import tempfile

import numpy as np

from .shared import TESTDATA_DIR, FILES

from ifem_to_vt import config
from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer

try:
    import vtfwriter
    has_vtf = True
except ImportError:
    has_vtf = False


NUM = re.compile(r'-?\d+[ +\-e.\d]*\n?$')


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname = request.param
    base, _ = splitext(basename(rootname))
    vtfname = '{}.vtf'.format(base)
    return (
        join(TESTDATA_DIR, rootdir, rootname),
        join(TESTDATA_DIR, 'vtf', vtfname),
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
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with config(output_mode='ascii'), get_reader(infile) as r, get_writer('vtf')(outfile) as w:
            r.write(w)
        assert exists(outfile)
        assert exists(checkfile)
        with open(outfile, 'r') as out, open(checkfile, 'r') as ref:
            compare_vtf(out, ref)
