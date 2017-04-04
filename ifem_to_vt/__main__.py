import click
from os.path import splitext
import sys

from ifem_to_vt.reader import Reader
from ifem_to_vt.writer import get_writer


@click.command()
@click.option('--fmt', '-f', type=click.Choice(['vtf']), required=False)
@click.argument('infile', type=str, required=True)
@click.argument('outfile', type=str, required=False)
def convert(fmt, infile, outfile):
    if outfile and not fmt:
        _, fmt = splitext(outfile)
        fmt = fmt[1:].lower()
    elif not outfile:
        fmt = fmt or 'vtf'
        basename, _ = splitext(infile)
        outfile = '{}.{}'.format(infile, fmt)

    try:
        Writer = get_writer(fmt)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    with Reader(infile) as r, Writer(outfile) as w:
        print(list(r.h5))


if __name__ == '__main__':
    convert()
