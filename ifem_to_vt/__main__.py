import click
import logging
from os.path import splitext, basename
import sys

from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer


@click.command()
@click.option(
    '--verbosity', '-v',
    type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
    default='info'
)
@click.option('--basis', '-b', multiple=True)
@click.option('--geometry', '-g', default=None)
@click.option('--fmt', '-f', type=click.Choice(['vtf']), required=False)
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)
def convert(verbosity, basis, geometry, fmt, infile, outfile):
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
        fmt = fmt or 'vtf'
        filename = basename(infile)
        base, _ = splitext(filename)
        outfile = '{}.{}'.format(base, fmt)

    try:
        Writer = get_writer(fmt)
    except ValueError as e:
        logging.critical(e)
        sys.exit(1)

    with get_reader(infile, bases=basis, geometry=geometry) as r, Writer(outfile) as w:
        r.write(w)


if __name__ == '__main__':
    convert()
