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
@click.option('--basis', '-b', multiple=True)
@click.option('--geometry', '-g', default=None)
@click.option('--nvis', '-n', default=1)
@click.option('--fmt', '-f', type=click.Choice(['vtf', 'vtk', 'vtu']), required=False)
@click.option('--mode', '-m', type=click.Choice(['binary', 'ascii', 'appended']), default='binary')
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)
@suppress_warnings
def convert(verbosity, basis, geometry, nvis, fmt, mode, infile, outfile):
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
        Writer = get_writer(fmt, mode=mode)
    except ValueError as e:
        logging.critical(e)
        sys.exit(1)

    reader_kwargs = {
        'bases': basis,
        'geometry': geometry,
        'nvis': nvis,
    }
    with get_reader(infile, **reader_kwargs) as r, Writer(outfile) as w:
        r.write(w)


if __name__ == '__main__':
    convert()
