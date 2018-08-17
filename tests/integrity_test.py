from os.path import dirname, join
import pytest
import tempfile

from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer


@pytest.fixture(params=[
    'Annulus',
    'Cavity-mixed', 'Cavity3D-compatible',
    'Cyl2D-VMSFSI-weak',
    'singular-pressure-corner-rec',
    'Square', 'Square-ad', 'Square-LR',
    'Square-compatible-abd1-B-I-stat', 'Square-mixed-abd1-B-I-stat',
    'Square-modes', 'Square-modes-freq',
    'Waterfall3D',
])
def filenames(request):
    path = join(dirname(__file__), 'testdata')
    fmt = lambda x: x.format(request.param)
    fn = request.param
    return join(path, fmt('{}.hdf5')), join(path, fmt('{}.vtf')), fmt('{}.vtf')


def test_integrity(filenames):
    infile, checkfile, outfile = filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)

        with get_reader(infile) as r, get_writer('vtf')(outfile) as w:
            r.write(w)
        with open(outfile, 'r') as out, open(checkfile, 'r') as ref:
            for outline, refline in zip(out, ref):
                if not 'EXPORT_DATE' in outline:
                    assert outline == refline
