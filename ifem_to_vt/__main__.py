from functools import wraps
import click
import logging
from os.path import splitext, basename
import sys
import warnings

from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer


def suppress_warnings(func):
    @wraps(func)
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Conversion of the second argument of issubdtype')
            return func(*args, **kwargs)
    return inner


@click.command()
@click.option(
    '--verbosity', '-v',
    type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
    default='info'
)
@click.option('--basis', '-b', multiple=True, help='Include fields in this basis.')
@click.option('--geometry', '-g', default=None, help='Use this basis to provide geometry.')
@click.option('--nvis', '-n', default=1, help='Extra sampling points per element.')
@click.option('--fmt', '-f', type=click.Choice(['vtf', 'vtk', 'vtu', 'pvd']), required=False, help='Output format.')
@click.option('--mode', '-m', type=click.Choice(['binary', 'ascii', 'appended']), default='binary', help='Output mode.')
@click.option('--last', is_flag=True, help='Read only the last step.')
@click.option('--endianness', type=click.Choice(['native', 'little', 'big']), default='native')
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)
@suppress_warnings
def convert(verbosity, basis, geometry, nvis, fmt, mode, last, endianness, infile, outfile):
    logging.basicConfig(
        format='{asctime} {levelname: <10} {message}',
        datefmt='%H:%M',
        style='{',
        level=verbosity.upper(),
    )

    if outfile and not fmt:
        _, fmt = splitext(outfile)
        fmt = fmt[1:].lower()
    elif not outfile:
        fmt = fmt or 'vtu'
        filename = basename(infile)
        base, _ = splitext(filename)
        outfile = '{}.{}'.format(base, fmt)

    try:
        Writer = get_writer(fmt)
    except ValueError as e:
        logging.critical(e)
        sys.exit(1)

    reader_kwargs = {
        'bases': basis,
        'geometry': geometry,
        'nvis': nvis,
        'last': last,
        'endianness': endianness,
    }

    writer_kwargs = {
        'last': last,
        'mode': mode,
    }

    with get_reader(infile, **reader_kwargs) as r, Writer(outfile, **writer_kwargs) as w:
        r.write(w)


if __name__ == '__main__':
    convert()
